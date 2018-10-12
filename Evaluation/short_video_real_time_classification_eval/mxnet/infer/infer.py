# -*- coding: utf-8 -*-

import random


class ServerApi(object):
    """
    统一算法预测接口类：
    **注**：
        1.handle为举办方验证接口，该接口必须返回预测分类list eg:[1, 3, 4]，参赛队伍需具体实现该接口
        2.模型装载操作必须在初始化方法中进行
        3.初始化方法必须提供gpu_id参数
        3.其他接口都为参考，可以选择实现或删除
    """
    def __init__(self, gpu_id=0):
        self.model = self.load_model(gpu_id)

    def video_frames(self, file_dir):
        """
        视频截帧
        :param file_dir: 视频路径
        :return:
        """
        return None

    def load_model(self, gpu_id):
        """
        模型装载
        :param gpu_id: 装载GPU编号
        :return:
        """
        return ''

    def predict(self, file_dir):
        """
        模型预测
        :param file_dir: 预测文件路径
        :return:
        """
        return None

    def handle(self, video_dir):
        """
        算法处理
        :param video_dir: 待处理单视频路径
        :return: 返回预测分类列表 eg:[1, 3, 4]
        """
        return random.sample(range(0, 60), 3)


