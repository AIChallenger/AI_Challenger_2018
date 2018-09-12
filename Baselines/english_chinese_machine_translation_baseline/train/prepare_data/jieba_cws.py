#!/bin/env python

import sys
import jieba

def jieba_cws(string):
    seg_list = jieba.cut(string.strip().decode('utf8'))
    return u' '.join(seg_list).encode('utf8')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.stderr.write('usage: %s + train.zh' % __file__)
        sys.exit(-1)
    filename = sys.argv[1]
    #fileout = open("%s.cws"%filename, 'wb')
    with open(filename, 'r') as f:
        for line in f:
            line_cws = jieba_cws(line)
            sys.stdout.write(line_cws.strip())
            sys.stdout.write('\n')
            #print line_cws.strip()
    

