# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:22.08-py3
ARG DEBIAN_FRONTEND=noninteractive

# apt install required packages
RUN apt update
RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 python3.8-dev default-libmysqlclient-dev build-essential \
    && apt install -y zip htop screen libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt

RUN pip install opencv-python-headless==4.1.2.30
RUN pip install -U albumentations albumentations_experimental
RUN python3 -m pip install git+https://github.com/sail-sg/Adan.git


#
RUN pip install -U typing_extensions
RUN pip install requests
RUN pip install -U numpy==1.22.0
RUN pip install onnxsim
RUN pip install thop


RUN apt-get update -y
RUN apt-get install -y libturbojpeg

RUN pip install -U git+https://github.com/lilohuang/PyTurboJPEG.git

WORKDIR /app
