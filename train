#!/usr/bin/env python3
from argparse import ArgumentParser
import gzip
import json
import torch
from torch.nn import Embedding, Module, ModuleList, Linear
from torch.nn import functional as F
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim import Adam
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import BatchNorm, GCNConv, global_add_pool

NODE_TYPES = 6
BATCH_SIZE = 64
CHANNELS = 8
CONVOLUTIONS = 8
HIDDEN = 128

class Graphs(InMemoryDataset):
    file: str
    def __init__(self, file: str):
        self.file = file
        super().__init__('.')
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> str:
        return self.file

    @property
    def processed_file_names(self):
        return self.file.replace('.jsonl.gz', '.pt')

    def process(self):
        data = []
        with gzip.open(self.file, 'r') as stream:
            for line in stream:
                raw = json.loads(line)
                data.append(Data(
                    x = torch.tensor(raw['nodes']),
                    edge_index = torch.tensor([
                        raw['sources'] + raw['targets'],
                        raw['targets'] + raw['sources']
                    ]),
                    y = torch.tensor(raw['y'], dtype=torch.float)
                ))
        data, slices = self.collate(data)
        torch.save((data, slices), self.processed_paths[0])

class Model(Module):
    embedding: Embedding
    bn: ModuleList
    conv: ModuleList
    hidden: Linear
    output: Linear

    def __init__(self):
        super().__init__()
        self.embedding = Embedding(NODE_TYPES, CHANNELS)
        self.bn = ModuleList([
            BatchNorm(CHANNELS)
            for _ in range(CONVOLUTIONS)
        ])
        self.conv = ModuleList([
            GCNConv(CHANNELS, CHANNELS)
            for _ in range(CONVOLUTIONS)
        ])
        self.hidden = Linear(CHANNELS, HIDDEN)
        self.output = Linear(HIDDEN, 1)

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        batch = batch.batch
        x = self.embedding(x)
        for bn, conv in zip(self.bn, self.conv):
            convolved = conv(bn(x), edge_index)
            x = F.relu(x + convolved)
        x = global_add_pool(x, batch)
        x = F.relu(self.hidden(x))
        return self.output(x).squeeze(-1)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('FILE')
    args = parser.parse_args()

    graphs = Graphs(args.FILE)
    model = Model().to('cuda')
    optimizer = Adam(model.parameters())

    step = 1
    writer = SummaryWriter()
    while True:
        for batch in DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=True):
            optimizer.zero_grad()
            batch = batch.to('cuda')
            prediction = model(batch)
            loss = F.binary_cross_entropy_with_logits(prediction, batch.y)
            loss.backward()
            optimizer.step()
            writer.add_scalar('loss', loss.detach(), global_step=step)
            if step % 100 == 0:
                print(torch.stack((torch.sigmoid(prediction).detach(), batch.y), dim=-1))
            step += 1
