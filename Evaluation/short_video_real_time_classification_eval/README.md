## AI-CHALLENGER相关文档

### Baseline
* 时间基准值 Time Baseline：**120ms/Video**
* 准确率基准值 ACC Baseline：0.800
* 说明：
    - 运行时间计算的是从读入视频路径到输出预测标签的所有时间
    - 最终计算排名使用的是误差Baseline，其值为1 - acc

### 测试机器配置
* GPU：
    - 1卡P40
* CPU：
    - 核心：8核（开了超线程的8核）
    - 内存：32G
    - 硬盘：300G

### 标准格式
#### 代码格式
* 参赛者需参考**infer.py**标准格式写好接口，不得改变**ServerApi**类名和**handle**函数名。
    - 其中**handle**函数的输入需为单视频路径，输出需为该视频识别多标签列表 eg:[1, 3, 4]
    - 参赛者需将模型加载等初始化操作并入__init__，并指定运行环境gpu_id。
    - 若参赛选手使用非python的其他语言，也需要使用python中转，实现infer.py的相关功能。
    
#### Dockerfile格式
* 不强制要求基础镜像的环境、框架和版本
* 具体参考：[Dockerfile](mxnet/Dockerfile)

#### 工作空间格式
* 必须指定/data为程序工作空间
    
#### 示例代码结构
* mxnet版本demo
    - infer：算法代码
    - Dockerfile：构建文档
    - requirements.txt：pip安装文件
    
```
.
├── Dockerfile
├── README.md
├── __init__.py
├── infer
│   ├── __init__.py
│   └── infer.py
└── requirements.txt
```


### 构建示例程序
* [mxnet版本构建样例](mxnet/README.md)


### 标签映射表

标签ID | 标签名称 | Tag Name | 标签ID | 标签名称 | Tag Name | 标签ID | 标签名称 | Tag Name |
--- | --- | --- | --- |--- | --- | --- | --- | --- |
0 | 狗 | Dog | 21 | 芭蕾舞 | Ballet | 42 | 游戏 | Games |
1 | 猫 | Cat | 22 | 广场舞 | Square Dancing | 43 | 娱乐 | Entertainment |
2 | 鼠 | Mouse | 23 | 民族舞 | Folk Dance | 44 | 动漫 | Animation |
3 | 兔子 | Rabbit | 24 | 绘画 | Drawing | 45 | 文字艺术配音 | Word Art Voicing |
4 | 鸟 | Bird | 25 | 手写文字 | Handwriting | 46 | 瑜伽 | Yoga |
5 | 风景 | Scenery | 26 | 咖啡拉花	 | Latte Art | 47 | 健身 | Fitness |
6 | 风土人情 | Local Customs | 27 | 沙画 | Sand Drawing | 48 | 滑板 | Skateboard |
7 | 穿秀 | Dressing | 28 | 史莱姆 | Slime | 49 | 篮球 | Basketball |
8 | 宝宝 | Baby | 29 | 折纸 | Origami | 50 | 跑酷 | Parkour |
9 | 男生自拍 | Selfie-Male | 30 | 编织 | Knitting | 51 | 潜水 | Diving |
10 | 女生自拍	 | Selfie-Female | 31 | 发饰 | Hair Accessory | 52 | 台球 | Billiards |
11 | 做甜品 | Dessert Making | 32 | 陶艺 | Pottery | 53 | 足球 | Football |
12 | 做海鲜 | Seafood Making | 33 | 手机壳 | Phone Case | 54 | 羽毛球 | Badminton |
13 | 街边小吃	 | Streetside Snacks | 34 | 打鼓 | Drums | 55 | 乒乓球 | Table Tennis |
14 | 饮品 | Drinks | 35 | 弹吉他 | Guitar | 56 | 画眉 | Brow Painting |
15 | 火锅 | Hot Pot | 36 | 弹钢琴 | Piano | 57 | 画眼 | Eyeliner |
16 | 抓娃娃 | Claw Crane | 37 | 弹古筝 | Guzheng | 58 | 护肤 | Skincare |
17 | 手势舞 | Handsign Dance | 38 | 拉小提琴 | Violin | 59 | 唇彩 | Lipgloss |
18 | 街舞 | Street Dance | 39 | 拉大提琴	| Cello | 60 | 卸妆 | Makeup Removal |
19 | 国标舞 | International Dance | 40 | 吹葫芦丝	 | Hulusi | 61 | 美甲 | Nail Cosmetic |
20 | 钢管舞 | Pole Dance | 41 | 唱歌 | Singing | 62 | 美发 | Hair Cosmetic |


