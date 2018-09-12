#!/bin/env python

import sys
import re

def extract_text(line):
    pattern = re.compile(r'<seg id=.*>(.*)</seg>')
    if pattern.search(line):
        line = pattern.search(line).group(1).strip()
        return line
    return False


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.stderr.write('usage: %s + input.sgm' % __file__)
        sys.exit(-1)
    filename = sys.argv[1]
    with open(filename, 'r') as f:
        for line in f:
            new_line = extract_text(line)
            if new_line:
                sys.stdout.write(new_line.strip()) 
                sys.stdout.write('\n')

