FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04
RUN apt-get update && \
    apt-get install -y software-properties-common curl git && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.6-dev && \
    curl https://bootstrap.pypa.io/get-pip.py | python3.6

RUN add-apt-repository ppa:git-core/ppa && \
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get update && \
    apt-get install git-lfs

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y build-essential python3.6-tk libgomp1 libsm6

WORKDIR /code

ADD requirements.txt .
RUN pip3 install -r requirements.txt

ADD . /code
