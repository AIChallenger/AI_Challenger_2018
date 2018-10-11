## AI-CHALLENGER相关文档

### 标准格式
#### 代码格式
* 参赛者需参考**infer.py**标准格式写好接口，不得改变**ServerApi**类名和**handle**函数名。
    - 其中**handle**函数的输入需为单视频路径，输出需为该视频识别标签。
    - 参赛者需将模型加载等初始化操作并入__init__，并指定运行环境gpu_id。
    - 若参赛选手使用非python的其他语言，也需要使用python中转，实现infer.py的相关功能。
    
#### Dockerfile格式
* 不强制要求基础镜像的环境、框架和版本
* 具体参考：[Dockerfile](mxnet/Dockerfile)

#### 工作空间格式
* 必须指定/data为程序工作空间
    
### 代码结构
* mxnet：mxnet版本demo
    - infer：算法代码
    - Dockerfile：构建文档
    - requirements.txt：pip安装文件
* data：测试文件夹
    - run.py：启动文件
	- video：测试数据集
    - input.txt：输入文件
* output: 输出文件夹
    - output.txt：输出文件
    - result.txt：结果文件
* <font color="red">注：data目录只做参考，后期会被组织方真实验证目录覆盖，故不能存放任何与算法相关的数据</font>

```
.
├── README.md
├── data
│   ├── input.txt
│   ├── run.py
│   ├── tag.txt
│   └── video
│       ├── 963193352.mp4
│       └── 970453214.mp4
├── mxnet
│   ├── Dockerfile
│   ├── README.md
│   ├── infer
│   │   └── infer.py
│   └── requirements.txt
└── output
    ├── output.txt
    └── result.txt
```

### 构建示例程序
* [mxnet版本构建样例](mxnet/README.md)


### 平台运行配置
* 文件入口配置：http://xxx.ufile.ucloud.com.cn/test/
* 文件输出配置：http://xxx.ufile.ucloud.com.cn/output/
* 启动入口配置：/data/data/run.py