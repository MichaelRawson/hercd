#!/usr/bin/env python3
import json
import random
from typing import List

from .cd import F, c
from .environment import Environment
from .graph import Graph

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
        print(environment.largest)
        if environment.proof is not None:
            return

def generate(axioms: List[F], goal: F):
    environment = Environment(axioms, goal)
    while True:
        environment.run()
        target, negatives, positives = environment.sample()
        for negative in negatives:
            output_sample(negative, target, False)
        for positive in positives:
            output_sample(positive, target, True)

if __name__ == '__main__':
    random.seed(0)
    AXIOMS: List[F] = [
        c(1,c(2,1)),
        c(c(c(1,'f'),'f'),1),
        c(c(1,c(2,3)),c(c(1,2),c(1,3)))
    ]
    GOAL: F = c(c(1,2),c(c(2,3),c(1,3)))

    import sys
    if sys.argv[1] == 'baseline':
        baseline(AXIOMS, GOAL)
    elif sys.argv[1] == 'generate':
        generate(AXIOMS, GOAL)
    else:
        print('not implemented: ', sys.argv[1])
