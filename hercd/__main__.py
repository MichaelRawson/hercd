#!/usr/bin/env python3
import atexit
import gzip
import json
import random
import torch

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard.writer import SummaryWriter

from .cd import F, n, c
from .constants import EPISODES, EXPERIENCE_BUFFER_LIMIT
from .environment import Environment
from .model import Model
from .train import BATCH_SIZE, CDDataset, forward, train_from_file

AXIOMS: list[F] = [
    c(c(c(c(c(1,2),c(n(3),n(4))),3),5),c(c(5,1),c(4,1)))
]

GOAL: F = c(c(1,2),c(c(2,3),c(1,3)))

STEPS: set[F] = {
    c(c(c(c(1,2),c(3,2)),c(2,4)),c(5,c(2,4))),
    c(c(c(1,c(n(2),3)),4),c(2,4)),
    c(c(c(1,1),2),c(3,2)),
    c(1,c(2,c(3,3))),
    c(c(c(1,c(2,2)),3),c(4,3)),
    c(c(c(1,2),3),c(2,3)),
    c(1,c(c(1,2),c(3,2))),
    c(1,c(c(c(2,1),3),c(4,3))),
    c(c(c(c(c(1,c(c(c(2,3),c(n(4),n(5))),4)),6),c(7,6)),2),c(5,2)),
    c(c(c(1,2),c(3,c(c(c(2,4),c(n(5),n(1))),5))),c(6,c(3,c(c(c(2,4),c(n(5),n(1))),5)))),
    c(1,c(c(c(2,3),4),c(c(c(3,5),c(n(4),n(2))),4))),
    c(c(c(1,2),3),c(c(c(2,4),c(n(3),n(1))),3)),
    c(c(c(1,2),c(n(c(c(c(3,1),4),c(5,4))),n(3))),c(c(c(3,1),4),c(5,4))),
    c(c(c(c(c(1,c(2,3)),4),c(5,4)),2),c(6,2)),
    c(c(c(1,2),c(n(c(3,1)),n(c(c(c(4,c(1,5)),6),c(7,6))))),c(3,1)),
    c(1,c(c(c(2,c(c(3,4),5)),4),c(3,4))),
    c(c(c(c(c(1,c(c(2,3),4)),3),c(2,3)),5),c(6,5)),
    c(c(c(1,2),c(3,c(c(n(2),n(4)),5))),c(4,c(3,c(c(n(2),n(4)),5)))),
    c(1,c(c(2,3),c(c(n(2),n(1)),3))),
    c(c(1,2),c(c(n(1),n(c(3,c(4,c(5,5))))),2)),
    c(c(n(c(c(1,c(2,2)),3)),n(c(4,c(5,c(6,6))))),c(7,3)),
    c(c(c(1,n(c(c(2,c(3,3)),n(1)))),4),c(5,4)),
    c(c(c(1,c(2,n(c(c(3,c(4,4)),n(2))))),5),c(6,5)),
    c(c(c(1,2),3),c(c(c(4,c(5,5)),n(n(2))),3)),
    c(c(c(1,c(2,2)),n(n(3))),c(c(3,4),c(5,4))),
    c(c(c(c(1,2),c(3,2)),4),c(n(n(1)),4)),
    c(c(c(c(c(c(1,2),c(3,2)),4),c(n(n(1)),4)),5),c(6,5)),
    c(c(c(1,n(2)),c(c(2,3),c(4,3))),c(5,c(c(2,3),c(4,3)))),
    c(c(c(c(c(1,2),c(3,2)),4),c(n(c(5,c(c(1,2),c(3,2)))),n(c(6,n(1))))),c(5,c(c(1,2),c(3,2)))),
    c(c(n(c(1,c(c(2,3),c(4,3)))),n(c(5,n(2)))),c(1,c(c(2,3),c(4,3)))),
    c(1,c(c(c(2,n(3)),c(4,5)),c(c(3,5),c(4,5)))),
    c(c(c(1,n(2)),c(3,4)),c(c(2,4),c(3,4))),
    c(c(1,c(2,3)),c(c(c(4,c(5,n(1))),3),c(2,3))),
    c(c(c(1,c(2,n(c(c(3,4),5)))),5),c(4,5)),
    c(c(c(1,2),3),c(c(c(4,1),2),3)),
    c(c(c(1,c(2,n(c(c(3,4),5)))),5),c(c(c(6,3),4),5)),
    c(c(c(c(c(1,2),3),4),5),c(c(c(2,3),4),5)),
    c(c(c(1,c(2,n(3))),c(4,5)),c(c(3,5),c(4,5))),
    c(c(1,c(2,3)),c(c(c(1,4),3),c(2,3))),
    c(c(c(1,2),c(c(n(3),n(1)),4)),c(c(3,4),c(c(n(3),n(1)),4))),
    c(c(1,2),c(c(n(1),n(c(2,3))),2)),
    c(c(c(1,c(2,n(c(3,4)))),4),c(c(n(3),n(c(4,5))),4)),
    c(c(c(c(n(1),n(c(2,3))),2),4),c(c(1,2),4)),
    c(c(c(c(n(1),n(2)),1),3),c(c(3,4),c(2,4))),
    c(c(c(n(1),2),3),c(c(3,4),c(1,4))),
    c(c(1,2),c(c(2,3),c(1,3)))
}

def baseline():
    """uniform-policy mode"""
    environment = Environment(AXIOMS, GOAL)
    environment.chatty = True
    writer = SummaryWriter()
    total_episodes = 0
    while True:
        environment.run()
        total_episodes += 1
        progress = sum(
            known.formula in STEPS
            for known in environment.known
        )
        writer.add_scalar('progress', progress, global_step=total_episodes)

        if environment.proof is not None:
            writer.add_scalar('steps to proof', len(environment.known), global_step=total_episodes)

def generate():
    """uniform-policy mode, but output training data"""
    environment = Environment(AXIOMS, GOAL)
    while True:
        environment.run()
        for major, minor, y in environment.training_graphs():
            print(json.dumps({
                'major': major.__dict__,
                'minor': minor.__dict__,
                'y': y
            }))

def learn():
    """online-learning 'reinforcement' mode"""

    model = Model().to('cuda')
    optimizer = Adam(model.parameters())
    writer = SummaryWriter()
    experience = []
    save: list[str] = []

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
        for _ in range(EPISODES):
            environment.run()
            total_episodes += 1

            progress = sum(
                known.formula in STEPS
                for known in environment.known
            )
            writer.add_scalar('progress', progress, global_step=total_episodes)

            if environment.proof is not None:
                writer.add_scalar('steps to proof', len(environment.known), global_step=total_episodes)

            for major, minor, y in environment.training_graphs():
                experience.append((major.torch(), minor.torch(), torch.tensor(float(y))))
                save.append(json.dumps({
                    'major': major.__dict__,
                    'minor': minor.__dict__,
                    'y': y
                }))

        random.shuffle(experience)
        while len(experience) > EXPERIENCE_BUFFER_LIMIT:
            experience.pop()

        environment.model = model
        model.train()
        dataset = CDDataset(experience)
        best_loss = float('inf')
        losses = []
        while True:
            for major, minor, y in DataLoader(dataset, collate_fn=CDDataset.collate, batch_size=BATCH_SIZE, shuffle=True):
                prediction, loss = forward(model, major, minor, y)
                if len(losses) == 0:
                    writer.add_histogram('prediction', prediction, global_step=total_batches)
                loss.backward()
                losses.append(float(loss))
                writer.add_scalar('loss', loss.detach(), global_step=total_batches)
                optimizer.step()
                optimizer.zero_grad()
                total_batches += 1

            new_loss = sum(losses) / len(losses)
            losses.clear()
            print(new_loss)
            if new_loss < best_loss:
                best_loss = new_loss
            else:
                break

if __name__ == '__main__':
    random.seed(0)
    torch.manual_seed(0)

    import sys
    if sys.argv[1] == 'baseline':
        baseline()
    elif sys.argv[1] == 'generate':
        generate()
    elif sys.argv[1] == 'train':
        train_from_file(sys.argv[2])
    elif sys.argv[1] == 'learn':
        learn()
    else:
        print('not implemented: ', sys.argv[1])
