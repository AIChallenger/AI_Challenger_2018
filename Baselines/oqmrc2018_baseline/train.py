# -*- coding: utf-8 -*-
import argparse
import cPickle
import torch

from model import MwAN
from preprocess import process_data
from utils import *

parser = argparse.ArgumentParser(description='PyTorch implementation for Multiway Attention Networks for Modeling '
                                             'Sentence Pairs of the AI-Challenges')

parser.add_argument('--data', type=str, default='data/',
                    help='location directory of the data corpus')
parser.add_argument('--threshold', type=int, default=5,
                    help='threshold count of the word')
parser.add_argument('--epoch', type=int, default=50,
                    help='training epochs')
parser.add_argument('--emsize', type=int, default=128,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=128,
                    help='hidden size of the model')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--log_interval', type=int, default=300,
                    help='# of batches to see the training error')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')

args = parser.parse_args()

# vocab_size = process_data(args.data, args.threshold)
vocab_size = 98745

model = MwAN(vocab_size=vocab_size, embedding_size=args.emsize, encoder_size=args.nhid, drop_out=args.dropout)
print('Model total parameters:', get_model_parameters(model))
if args.cuda:
    model.cuda()
optimizer = torch.optim.Adamax(model.parameters())

with open(args.data + 'train.pickle', 'rb') as f:
    train_data = cPickle.load(f)
with open(args.data + 'dev.pickle', 'rb') as f:
    dev_data = cPickle.load(f)
dev_data = sorted(dev_data, key=lambda x: len(x[1]))

print('train data size {:d}, dev data size {:d}'.format(len(train_data), len(dev_data)))


def train(epoch):
    model.train()
    data = shuffle_data(train_data, 1)
    total_loss = 0.0
    for num, i in enumerate(range(0, len(data), args.batch_size)):
        one = data[i:i + args.batch_size]
        query, _ = padding([x[0] for x in one], max_len=50)
        passage, _ = padding([x[1] for x in one], max_len=350)
        answer = pad_answer([x[2] for x in one])
        query, passage, answer = torch.LongTensor(query), torch.LongTensor(passage), torch.LongTensor(answer)
        if args.cuda:
            query = query.cuda()
            passage = passage.cuda()
            answer = answer.cuda()
        optimizer.zero_grad()
        loss = model([query, passage, answer, True])
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
        if (num + 1) % args.log_interval == 0:
            print '|------epoch {:d} train error is {:f}  eclipse {:.2f}%------|'.format(epoch,
                                                                                         total_loss / args.log_interval,
                                                                                         i * 100.0 / len(data))
            total_loss = 0


def test():
    model.eval()
    r, a = 0.0, 0.0
    with torch.no_grad():
        for i in range(0, len(dev_data), args.batch_size):
            one = dev_data[i:i + args.batch_size]
            query, _ = padding([x[0] for x in one], max_len=50)
            passage, _ = padding([x[1] for x in one], max_len=500)
            answer = pad_answer([x[2] for x in one])
            query, passage, answer = torch.LongTensor(query), torch.LongTensor(passage), torch.LongTensor(answer)
            if args.cuda:
                query = query.cuda()
                passage = passage.cuda()
                answer = answer.cuda()
            output = model([query, passage, answer, False])
            r += torch.eq(output, 0).sum().item()
            a += len(one)
    return r * 100.0 / a


def main():
    best = 0.0
    for epoch in range(args.epoch):
        train(epoch)
        acc = test()
        if acc > best:
            best = acc
            with open(args.save, 'wb') as f:
                torch.save(model, f)
        print 'epcoh {:d} dev acc is {:f}, best dev acc {:f}'.format(epoch, acc, best)


if __name__ == '__main__':
    main()
