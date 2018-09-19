#!/bin/env python

import re
import subprocess
import os,sys
import argparse

def parse_args():
    p = argparse.ArgumentParser("BLEU/NIST/TER score for Machine Translation system")
    p.add_argument('-rs', '--ref-sgm', required=True, help="Reference file in sgm format")
    p.add_argument('-hs', '--hyp-sgm', required=True, help="Hypothesis file in sgm format")
    p.add_argument('-ss', '--src-sgm', required=True, help="Source file in sgm format")
    p.add_argument('-i', '--id', required=True, help="ID for this run")
    return p.parse_args()

def check(args):
    check_pass = True
    if not os.path.exists(args.ref_sgm):
        check_pass = False
        print >>sys.stderr, "ERROR: not find %s"%args.ref_sgm
    if not os.path.exists(args.src_sgm):
        check_pass = False
        print >>sys.stderr, "ERROR: not find %s"%args.src_sgm
    if not os.path.exists(args.hyp_sgm):
        check_pass = False
        print >>sys.stderr, "ERROR: not find %s"%args.hyp_sgm
    return check_pass

if __name__ == "__main__":
    # check requirments
    bin=os.path.dirname(__file__)
    args = parse_args()

    # check args
    if not check(args):
        print >> sys.stderr, "ERROR: check failed\n"
        sys.exit(1)

    work_dir = "work-"+args.id
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    # calc BLEU/NIST score
    os.system( "%s/chi_char_segment.pl -type xml < %s > %s/hyp.seg.sgm"%(bin, args.hyp_sgm, work_dir) )
    os.system( "%s/chi_char_segment.pl -type xml < %s > %s/ref.seg.sgm"%(bin, args.ref_sgm, work_dir) )
    os.system("%s/mteval-v11b.pl -s %s -r %s/ref.seg.sgm -t %s/hyp.seg.sgm -c > %s/bleu " % (bin, args.src_sgm, work_dir, work_dir,  work_dir) )

    # get bleu: NIST score = 5.5073  BLEU score = 0.2902 for system "DemoSystem"
    try:
        raw_bleu = " ".join(open(work_dir + "/bleu", 'r').readlines()).replace("\n"," ")
        gps = re.search( r'NIST score = (?P<NIST>[\d\.]+)  BLEU score = (?P<BLEU>[\d\.]+) ' , raw_bleu )
        if gps:
            bleu_score = gps.group('BLEU')
        else:
            print >>sys.stderr, "ERROR: unable to get bleu and nist score"
            sys.exit(1)
    except:
        print >>sys.stderr, "ERROR: exception during calculating bleu score"

    # sum score
    print "BLEU score=%s"%(bleu_score)



