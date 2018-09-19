## Description ##

    Evaluation utils for both Machine Simulation Interpretation and Text Machine Translation tasks.

## Requirements ##
- python 2.7
- perl 5

## Steps ##

Step 1: wrap your translation result into sgm file
```
[ ! -s work-demo1 ] && mkdir work-demo1
./tools/wrap_xml.pl zh data1/src.sgm DemoSystem < data1/hyp > work-demo1/hyp.sgm
```

Step 2: Segment reference and hyp sgm files
```
./tools/chi_char_segment.pl -t xml < work-demo1/hyp.sgm > work-demo1/hyp.seg.sgm 
./tools/chi_char_segment.pl -t xml < data1/ref.sgm > work-demo1/ref.seg.sgm 
```

Step 3: Calculate BLEU score for the translation result
```
./tools/mteval-v11b.pl -s data1/src.sgm -r work-demo1/ref.seg.sgm -t work-demo1/hyp.seg.sgm -c > work-demo1/bleu.log
```

## References ##

- BLEU: [BLEU: a Method for Automatic Evaluation of Machine Translation](http://www.aclweb.org/anthology/P02-1040.pdf)


## Upload File Formation For AI Challenger Competition ##
.txt, .sgm, or no suffix, such as 'output.txt' or 'output.sgm' or 'output'


