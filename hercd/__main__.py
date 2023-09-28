#!/usr/bin/env python3
import random
import torch

from torch.optim import Adam
from torch.utils.tensorboard.writer import SummaryWriter

from .cd import F, c
from .constants import BATCHES_PER_EPISODE, MAX_EXPERIENCE
from .environment import Environment
from .graph import Graph
from .model import Model
from .train import BATCH_SIZE, compute_loss, train_from_file

def baseline(axioms: list[F], goal: F):
    """uniform-policy mode"""
    environment = Environment(axioms, goal)
    while True:
        environment.run()
        if environment.proof is not None:
            return

def generate(axioms: list[F], goal: F):
    """uniform-policy mode, but output training data"""
    environment = Environment(axioms, goal)
    environment.silent = True
    while True:
        environment.run()
        for graph in environment.training_graphs():
            print(graph.json())

def learn(axioms: list[F], goal: F):
    """online-learning 'reinforcement' mode"""

    model = Model().to('cuda')
    optimizer = Adam(model.parameters())
    writer = SummaryWriter()
    experience = []

    environment = Environment(axioms, goal)
    environment.model = model
    total_episodes = 0
    total_batches = 0
    while True:
        environment.run()
        total_episodes += 1
        if environment.proof is not None:
            writer.add_scalar('steps to proof', len(environment.known), global_step=total_episodes)

        experience.extend((graph.torch() for graph in environment.training_graphs()))
        random.shuffle(experience)
        while len(experience) > MAX_EXPERIENCE:
            experience.pop()

        model.train()
        episode_batches = 0
        graphs_in_batch = 0
        while episode_batches < BATCHES_PER_EPISODE:
            for graph in experience:
                if episode_batches >= BATCHES_PER_EPISODE:
                    break
                loss = compute_loss(model, graph)
                loss.backward()
                graphs_in_batch += 1
                if graphs_in_batch == BATCH_SIZE:
                    writer.add_scalar('loss', loss.detach(), global_step=total_batches)
                    graphs_in_batch = 0
                    episode_batches += 1
                    total_batches += 1
                    optimizer.step()
                    optimizer.zero_grad()


AXIOMS: list[F] = [
    c(1,c(2,1)),
    c(c(c(1,'*'),'*'),1),
    c(c(1,c(2,3)),c(c(1,2),c(1,3)))
]
GOAL: F = c(c(1,2),c(c(2,3),c(1,3)))
#c(c(c(c(c(1,2),c(3,'*')),4),'*'),c(c('*',1),c(3,1)))


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
