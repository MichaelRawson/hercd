#!/usr/bin/env python3
import json
import random
import torch
from typing import List

from torch.optim import Adam
from torch.utils.tensorboard.writer import SummaryWriter
from torch_geometric.loader import DataLoader

from .cd import F, c
from .constants import BATCHES_PER_EPISODE, MAX_EXPERIENCE
from .environment import Environment
from .graph import Graph
from .model import Model
from .train import BATCH_SIZE, train_from_file, train_step

def output_sample(sample: F, target: F, y: bool):
    graph = Graph(sample, target)
    print(json.dumps({
        'nodes': graph.nodes,
        'sources': graph.sources,
        'targets': graph.targets,
        'y': y
    }))

def baseline(axioms: List[F], goal: F):
    environment = Environment(axioms, goal)
    while True:
        environment.run()
        if environment.proof is not None:
            return

def generate(axioms: List[F], goal: F):
    environment = Environment(axioms, goal)
    while True:
        environment.run()
        target, negatives, positives = environment.data()
        for negative in negatives:
            output_sample(negative, target, False)
        for positive in positives:
            output_sample(positive, target, True)

def learn(axioms: List[F], goal: F):
    model = Model().to('cuda')
    optimizer = Adam(model.parameters())
    writer = SummaryWriter()
    experience = []

    environment = Environment(axioms, goal, model)
    total_episodes = 0
    total_batches = 0
    while True:
        environment.run()
        target, negatives, positives = environment.data()

        total_episodes += 1
        writer.add_scalar('positive', len(positives), global_step=total_episodes)
        writer.add_scalar('negative', len(negatives), global_step=total_episodes)
        print(f'target: {target}, +ve: {len(positives)}, -ve: {len(negatives)}')
        if environment.proof is not None:
            writer.add_scalar('steps to proof', len(environment.known), global_step=total_episodes)

        for negative in negatives:
            experience.append(Graph(negative, target).torch(False))
        for positive in positives:
            experience.append(Graph(positive, target).torch(True))
        random.shuffle(experience)
        while len(experience) > MAX_EXPERIENCE:
            experience.pop()

        model.train()
        episode_batches = 0
        while episode_batches < BATCHES_PER_EPISODE:
            loader = DataLoader(experience, batch_size=BATCH_SIZE, shuffle=True)
            for batch in loader:
                episode_batches += 1
                total_batches += 1
                if episode_batches >= BATCHES_PER_EPISODE:
                    break
                _, loss = train_step(model, optimizer, batch)
                writer.add_scalar('loss', loss, global_step=total_batches)

AXIOMS: List[F] = [
    c(1,c(2,1)),
    c(c(c(1,'*'),'*'),1),
    c(c(1,c(2,3)),c(c(1,2),c(1,3)))
]
GOAL: F = c(c(c(c(c(1,2),c(3,'*')),4),'*'),c(c('*',1),c(3,1)))
#c(c(1,2),c(c(2,3),c(1,3)))

if __name__ == '__main__':
    random.seed(0)
    torch.manual_seed(0)

    import sys
    if sys.argv[1] == 'baseline':
        baseline(AXIOMS, GOAL)
    elif sys.argv[1] == 'generate':
        generate(AXIOMS, GOAL)
    elif sys.argv[1] == 'train':
        train_from_file(sys.argv[2])
    elif sys.argv[1] == 'learn':
        learn(AXIOMS, GOAL)
    else:
        print('not implemented: ', sys.argv[1])
