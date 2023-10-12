#!/usr/bin/env python3
import json
import random
import torch

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard.writer import SummaryWriter

from .cd import F, c
from .constants import BATCHES, EPISODES, EXPERIENCE_BUFFER_LIMIT
from .environment import Environment
from .model import Model
from .train import BATCH_SIZE, CDDataset, forward, train_from_file

AXIOMS: list[F] = [
    c(1,c(2,1)),
    c(c(c(1,'*'),'*'),1),
    c(c(1,c(2,3)),c(c(1,2),c(1,3)))
]

GOAL: F = c(c(1,2),c(c(2,3),c(1,3)))

STEPS: set[F] = {
    c(c(1,c(2,c(c(3,2),4))),c(1,c(2,4))),
    c(c(c(c(1,c(2,3)),c(1,2)),c(1,c(2,3))),c(c(c(1,c(2,3)),c(1,2)),c(1,3))),
    c(c(c(1,c(2,3)),c(1,2)),c(c(2,3),c(1,3))),
    c(c(1,c(2,3)),c(c(1,2),c(1,3))),
    c(1,c(c(2,c(c(3,2),4)),c(2,4))),
    c(c(c(1,c(2,c(3,2))),c(1,2)),c(1,c(3,2))),
    c(c(c(1,c(2,3)),c(1,2)),c(c(1,c(2,3)),c(1,3))),
    c(1,c(2,c(3,2))),
    c(c(1,c(c(2,1),3)),c(1,3)),
    c(c(c(1,c(2,3)),c(1,2)),c(4,c(c(1,c(2,3)),c(1,3)))),
    c(c(1,2),c(c(2,3),c(1,3))),
    c(1,c(c(c(2,c(3,4)),c(2,3)),c(c(2,c(3,4)),c(2,4)))),
    c(c(c(c(1,c(2,3)),c(1,2)),c(c(c(1,c(2,3)),c(1,3)),4)),c(c(c(1,c(2,3)),c(1,2)),4)),
    c(1,c(2,1)),
    c(1,c(2,c(3,c(4,3)))),
    c(c(c(1,2),3),c(2,3))
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

            experience.extend((
                (major.torch(), minor.torch(), torch.tensor(float(y))))
                for major, minor, y in environment.training_graphs()
            )

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
