# -*- coding: utf-8 -*-
import cPickle
import json

import jieba


def seg_line(line):
    return list(jieba.cut(line))


def seg_data(path):
    print 'start process ', path
    data = []
    with open(path, 'r') as f:
        for line in f:
            dic = json.loads(line, encoding='utf-8')
            question = dic['query']
            doc = dic['passage']
            alternatives = dic['alternatives']
            data.append([seg_line(question), seg_line(doc), alternatives.split('|'), dic['query_id']])
    return data


def build_word_count(data):
    wordCount = {}

    def add_count(lst):
        for word in lst:
            if word not in wordCount:
                wordCount[word] = 0
            wordCount[word] += 1

    for one in data:
        [add_count(x) for x in one[0:3]]
    print 'word type size ', len(wordCount)
    return wordCount


def build_word2id(wordCount, threshold=10):
    word2id = {'<PAD>': 0, '<UNK>': 1}
    for word in wordCount:
        if wordCount[word] >= threshold:
            if word not in word2id:
                word2id[word] = len(word2id)
        else:
            chars = list(word)
            for char in chars:
                if char not in word2id:
                    word2id[char] = len(word2id)
    print 'processed word size ', len(word2id)
    return word2id


def transform_data_to_id(raw_data, word2id):
    data = []

    def map_word_to_id(word):
        output = []
        if word in word2id:
            output.append(word2id[word])
        else:
            chars = list(word)
            for char in chars:
                if char in word2id:
                    output.append(word2id[char])
                else:
                    output.append(1)
        return output

    def map_sent_to_id(sent):
        output = []
        for word in sent:
            output.extend(map_word_to_id(word))
        return output

    for one in raw_data:
        question = map_sent_to_id(one[0])
        doc = map_sent_to_id(one[1])
        candidates = [map_word_to_id(x) for x in one[2]]
        length = [len(x) for x in candidates]
        max_length = max(length)
        if max_length > 1:
            pad_len = [max_length - x for x in length]
            candidates = [x[0] + [0] * x[1] for x in zip(candidates, pad_len)]
        data.append([question, doc, candidates, one[-1]])
    return data


def process_data(data_path, threshold):
    train_file_path = data_path + 'ai_challenger_oqmrc_validationset_20180816/ai_challenger_oqmrc_validationset.json'
    dev_file_path = data_path + 'ai_challenger_oqmrc_trainingset_20180816/ai_challenger_oqmrc_trainingset.json'
    test_a_file_path = data_path + 'ai_challenger_oqmrc_testa_20180816/ai_challenger_oqmrc_testa.json'
    test_b_file_path = data_path + 'ai_challenger_oqmrc_testb_20180816/ai_challenger_oqmrc_testb.json'
    path_lst = [train_file_path, dev_file_path, test_a_file_path, test_b_file_path]
    output_path = [data_path + x for x in ['dev.pickle', 'train.pickle', 'testa.pickle', 'testb.pickle']]
    return _process_data(path_lst, threshold, output_path)


def _process_data(path_lst, word_min_count=5, output_file_path=[]):
    raw_data = []
    for path in path_lst:
        raw_data.append(seg_data(path))
    word_count = build_word_count([y for x in raw_data for y in x])
    with open('data/word-count.obj', 'wb') as f:
        cPickle.dump(word_count, f)
    word2id = build_word2id(word_count, word_min_count)
    with open('data/word2id.obj', 'wb') as f:
        cPickle.dump(word2id, f)
    for one_raw_data, one_output_file_path in zip(raw_data, output_file_path):
        with open(one_output_file_path, 'wb') as f:
            one_data = transform_data_to_id(one_raw_data, word2id)
            cPickle.dump(one_data, f)
    return len(word2id)
