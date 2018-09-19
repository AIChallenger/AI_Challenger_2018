# run normally
# data/ref.sgm 是参看答案，比赛的时候会提供参看答案
# data/hyp.sgm 是选手上传的翻译结果，打包成sgm文件
# data/src.sgm 是原始需要翻译的文件，sgm格式
# --id 标识此次计算得分过程。程序会为不同的id存不同的目录。
[ ! -s score ] && mkdir score
./tools/mt-score-main.py -rs data1/ref.sgm -hs data1/hyp.sgm -ss data1/src.sgm --id demo1 | tee score/demo1.score
./tools/mt-score-main.py -rs data2/ref.sgm -hs data2/hyp.sgm -ss data2/src.sgm --id demo2 | tee score/demo2.score
./tools/mt-score-main.py -rs data3/ref.sgm -hs data3/hyp.sgm -ss data3/src.sgm --id demo3 | tee score/demo3.score

# missing file
# 丢失文件的情况
./tools/mt-score-main.py -rs data/ref.sgm -hs data/hyp.sgm -ss data/src_.sgm --id demo


