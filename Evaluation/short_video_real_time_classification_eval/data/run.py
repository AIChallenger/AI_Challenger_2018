# -*- coding:utf-8 -*-


import os
import time
import json
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

DATA_DIR = os.path.join(ROOT_DIR, 'data')
VIDEO_DIR = os.path.join(DATA_DIR, 'video')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')

from infer.infer import ServerApi

server = ServerApi(0)


def verify_method(type):
    start_time = time.time()
    # 视频读取
    video_list = []
    with open(os.path.join(DATA_DIR, 'input_%s.txt' % type), 'r+') as f:
        line = f.readline()
        while line:
            video_path = line.replace('\n', '')
            video_list.append(os.path.join(VIDEO_DIR, video_path))
            line = f.readline()
    # 视频处理
    result_list = []
    for video in video_list:
        video_name = video.split('/')[-1]
        try:
            cla = int(server.handle(video))
        except Exception:
            cla = -1        # -1代表预测异常
        result_list.append([video_name, cla])
    # 标签输出
    with open(os.path.join(OUTPUT_DIR, 'output_%s.txt' % type), 'w+') as f:
        for result in result_list:
            out_info = str(result[0]) + ',' + str(result[1]) + '\n'
            f.write(out_info)
    end_time = time.time()
    return len(video_list), end_time - start_time, result_list


def time_counter():
    """
    处理耗时统计
    :return:    平均耗时
    """
    N = 10          # 遍历次数
    num = 0         # 文本数
    total = 0.0     # 总耗时
    time_list = []
    for i in range(N):
        count, t_val, result_list = verify_method('time')
        num = count
        time_list.append(t_val)
    time_list.sort()

    for item in time_list[1: -1]:
        total += item
    return total / (num * (N - 2))


def acc_counter():
    """
    准确率统计
    :return:    准确率
    """
    total, t_val, result_list = verify_method('acc')
    label_list = []
    with open(os.path.join(DATA_DIR, 'tag.txt'), 'r+') as f:
        line = f.readline()
        while line:
            line = line.replace('\n', '')
            label_list.append(line.split(',')[1])
            line = f.readline()
    result_list = list(zip(* result_list)[1])
    num = 0
    for i, val in enumerate(result_list):
        if int(val) == int(label_list[i]):
            num += 1
    return num / float(total)


if __name__ == '__main__':
    BASELINE_TIME = 0.1
    result = {
        "time": 0,
        "acc_rate": 0.0
    }
    avg_time = time_counter()
    result['time'] = avg_time
    if avg_time <= BASELINE_TIME:       # 速度不达标，不进行后续验证
        acc_rate = acc_counter()
        result['acc_rate'] = acc_rate
    # 结果保存
    with open(os.path.join(OUTPUT_DIR, 'result.txt'), 'w+') as f:
        f.write(json.dumps(result))