#!/usr/bin/env python3
import atexit
import gc
import gzip
import json
import random

import torch
from torch.utils.data import random_split
from torch.utils.tensorboard.writer import SummaryWriter

from .cd import Entry, F, n, c, D
from .constants import EPISODES_PER_EPOCH, EXPERIENCE_BUFFER_LIMIT, SAMPLES_PER_EPISODE
from .environment import Environment
from .graph import Graph
from .model import Model
from .train import CDDataset, create_optimizer, validate, validate_steps, epoch

AXIOMS: list[Entry] = [Entry(c(c(c(c(c(1,2),c(n(3),n(4))),3),5),c(c(5,1),c(4,1))))]

GOAL: F = c(c(1,2),c(c(2,3),c(1,3)))

S1 = AXIOMS[0]
S2a = D(S1, S1)
S2b = D(S1, S2a)
S2 = D(S1, S2b)
S3 = D(S2, S2)
S4 = D(S1, S3)
S5 = D(S1, S4)
S6 = D(S5, S1)
S7 = D(S5, S6)
S8a = D(S1, S7)
S8b = D(S1, S8a)
S8c = D(S8b, S6)
S8 = D(S8c, S1)
S9 = D(S8, S6)
S10a = D(S1, S9)
S10 = D(S8, S10a)
S11a = D(S4, S10)
S11b = D(S1, S11a)
S11c = D(S1, S11b)
S11 = D(S11c, S1)
S12a = D(S11, S3)
S12b = D(S12a, S4)
S12c = D(S9, S12b)
S12d = D(S9, S12c)
S12e = D(S1, S12d)
S12f = D(S12e, S1)
S12g = D(S1, S12f)
S12h = D(S6, S12g)
S12 = D(S1, S12h)
S13a = D(S8, S12)
S13b = D(S5, S13a)
S13c = D(S12, S13b)
S13d = D(S13c, S1)
S13 = D(S13d, S7)
S14a = D(S13, S5)
S14b = D(S1, S14a)
S14c = D(S13, S14b)
S14 = D(S1, S14c)
S15a = D(S13, S6)
S15b = D(S15a, S9)
S15c = D(S15b, S11)
S15d = D(S15c, S10)
S15e = D(S13, S15d)
S15f = D(S1, S15e)
S15g = D(S14, S1)
S15h = D(S14, S15g)
S15 = D(S15f, S15h)

DEDUCTION: list[Entry] = [
    S2a,
    S2b,
    S2,
    S3,
    S4,
    S5,
    S6,
    S7,
    S8a,
    S8b,
    S8c,
    S8,
    S9,
    S10a,
    S10,
    S11a,
    S11b,
    S11c,
    S11,
    S12a,
    S12b,
    S12c,
    S12d,
    S12e,
    S12f,
    S12g,
    S12h,
    S12,
    S13a,
    S13b,
    S13c,
    S13d,
    S13,
    S14a,
    S14b,
    S14c,
    S14,
    S15a,
    S15b,
    S15c,
    S15d,
    S15e,
    S15f,
    S15g,
    S15h,
    S15
]

STEPS: set[F] = {entry.formula for entry in DEDUCTION}
STEP_DATASET = CDDataset([Graph(entry, GOAL).torch() for entry in DEDUCTION])

def baseline():
    """uniform-policy mode"""
    environment = Environment(AXIOMS, GOAL)
    environment.chatty = True
    writer = SummaryWriter()
    total_episodes = 0
    while True:
        environment.run()
        total_episodes += 1
        progress = sum(entry.formula in STEPS for entry in environment.active + environment.passive)
        writer.add_scalar('proof/progress', progress, global_step=total_episodes)

def generate():
    """uniform-policy mode, but output training data"""
    environment = Environment(AXIOMS, GOAL)
    while True:
        environment.run()
        for _ in range(SAMPLES_PER_EPISODE):
            target, positive, negative = environment.sample()
            for y, sample in (1.0, positive), (0.0, negative):
                graph = Graph(sample, target)
                graph.y = y
                print(json.dumps(graph.__dict__, default=list))


def train(path: str):
    """offline-train a model from data provided in `path`"""

    dataset = CDDataset.from_file(path)
    train, test = random_split(dataset, [.95, .05])
    model = Model().to('cuda')
    optimizer = create_optimizer(model)

    step = 1
    writer = SummaryWriter()
    while True:
        step = epoch(model, optimizer, train, writer, step)
        validate(model, test, writer, step)
        validate_steps(model, STEP_DATASET, writer, step)


def learn():
    """online-learning 'reinforcement' mode"""

    model = Model().to('cuda')
    optimizer = create_optimizer(model)
    writer = SummaryWriter()
    experience = []
    save = []

    def save_on_exit():
        print(f'saving {len(save)} data to save.jsonl.gz...')
        with gzip.open('save.jsonl.gz', 'w') as f:
            for line in save:
                f.write(line.encode('ascii'))
                f.write(b'\n')
    atexit.register(save_on_exit)

    environment = Environment(AXIOMS, GOAL)
    environment.chatty = True
    total_episodes = 0
    total_batches = 0

    while True:
        for _ in range(EPISODES_PER_EPOCH):
            environment.run()
            total_episodes += 1

            progress = sum(entry.formula in STEPS for entry in environment.active + environment.passive)
            writer.add_scalar('proof/progress', progress, global_step=total_episodes)

            for _ in range(SAMPLES_PER_EPISODE // 2):
                target, positive, negative = environment.sample()
                for y, sample in (1.0, positive), (0.0, negative):
                    graph = Graph(sample, target)
                    graph.y = y
                    experience.append(graph.torch())
                    save.append(json.dumps(graph.__dict__, default=list))

        random.shuffle(experience)
        random.shuffle(save)
        while len(experience) > EXPERIENCE_BUFFER_LIMIT:
            experience.pop()
            save.pop()
        gc.collect()

        environment.model = model
        dataset = CDDataset(experience)
        train, test = random_split(dataset, [.95, .05])
        total_batches = epoch(model, optimizer, train, writer, total_batches)
        validate(model, test, writer, total_batches)
        validate_steps(model, STEP_DATASET, writer, total_batches)

if __name__ == '__main__':
    random.seed(0)
    torch.manual_seed(0)

    import sys
    if sys.argv[1] == 'baseline':
        baseline()
    elif sys.argv[1] == 'generate':
        generate()
    elif sys.argv[1] == 'train':
        train(sys.argv[2])
    elif sys.argv[1] == 'learn':
        learn()
    else:
        print('not implemented: ', sys.argv[1])
