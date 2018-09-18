# -*- coding: utf-8 -*-
import argparse
import cPickle
import codecs

import torch
from utils import *

from preprocess import seg_data, transform_data_to_id

parser = argparse.ArgumentParser(description='inference procedure, note you should train the data at first')

parser.add_argument('--data', type=str,
                    default='data/ai_challenger_oqmrc_testa_20180816/ai_challenger_oqmrc_testa.json',
                    help='location of the test data')

parser.add_argument('--word_path', type=str, default='data/word2id.obj',
                    help='location of the word2id.obj')

parser.add_argument('--output', type=str, default='data/prediction.a.txt',
                    help='prediction path')
parser.add_argument('--model', type=str, default='model.pt',
                    help='model path')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--cuda', action='store_true',default=True,
                    help='use CUDA')

args = parser.parse_args()

with open(args.model, 'rb') as f:
    model = torch.load(f)
if args.cuda:
    model.cuda()

with open(args.word_path, 'rb') as f:
    word2id = cPickle.load(f)

raw_data = seg_data(args.data)
transformed_data = transform_data_to_id(raw_data, word2id)
data = [x + [y[2]] for x, y in zip(transformed_data, raw_data)]
data = sorted(data, key=lambda x: len(x[1]))
print 'test data size {:d}'.format(len(data))


def inference():
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(0, len(data), args.batch_size):
            one = data[i:i + args.batch_size]
            query, _ = padding([x[0] for x in one], max_len=50)
            passage, _ = padding([x[1] for x in one], max_len=300)
            answer = pad_answer([x[2] for x in one])
            str_words = [x[-1] for x in one]
            ids = [x[3] for x in one]
            query, passage, answer = torch.LongTensor(query), torch.LongTensor(passage), torch.LongTensor(answer)
            if args.cuda:
                query = query.cuda()
                passage = passage.cuda()
                answer = answer.cuda()
            output = model([query, passage, answer, False])
            for q_id, prediction, candidates in zip(ids, output, str_words):
                prediction_answer = u''.join(candidates[prediction])
                predictions.append(str(q_id) + '\t' + prediction_answer)
    outputs = u'\n'.join(predictions)
    with codecs.open(args.output, 'w',encoding='utf-8') as f:
        f.write(outputs)
    print 'done!'


if __name__ == '__main__':
    inference()
