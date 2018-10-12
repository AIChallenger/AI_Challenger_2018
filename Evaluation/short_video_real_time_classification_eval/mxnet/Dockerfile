FROM mxnet/python:gpu_0.11.0
LABEL maintainer "xxx@meitu.com"

# install python
RUN rm -rf /var/lib/apt/lists/*
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
            aptitude git vim make wget zip zsh pkg-config \
            build-essential checkinstall p7zip-full python-pip \
            python3-pip tmux ffmpeg i7z unrar htop cmake g++  \
            curl libopenblas-dev python-numpy python3-numpy \
            python python-tk idle python-pmw python-imaging \
            libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev \
            libtbb2 libtbb-dev  libdc1394-22-dev libavcodec-dev  \
            libavformat-dev libswscale-dev libv4l-dev libatlas-base-dev \
            gfortran && \
    apt-get autoremove && \
    apt-get clean && \
    aptitude install -y python-dev && \
    # update pip and setuptools
    pip install --upgrade pip setuptools


###############################################
# 以上部分用户可定制，以下部分不可删除
###############################################
# 项目构建
WORKDIR /data
COPY . .

# 依赖软件安装
RUN pip install -r requirements.txt -i http://pypi.douban.com/simple  --trusted-host pypi.douban.com

