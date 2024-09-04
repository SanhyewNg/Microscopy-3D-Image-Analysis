FROM ubuntu:16.04
RUN apt-get update && \
    apt-get install -y software-properties-common curl git && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.6-dev && \
    curl https://bootstrap.pypa.io/get-pip.py | python3.6

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

RUN apt-get update && \
    apt-get install -y \
        build-essential \
        python3.6-tk \
        libgomp1 \
        libsm6 \
        openjdk-8-jdk \
        gnuplot \
        zip

WORKDIR /code
COPY ./models/model_7* /code/models/
COPY ./models/model_8* /code/models/
COPY ./models/classification /code/models/classification
ADD requirements.txt /code/
RUN sed -i -- "s|tensorflow-gpu|tensorflow|g" /code/requirements.txt
RUN pip3 install -r /code/requirements.txt --no-cache-dir
ADD ./clb /code/clb/
ADD ./vendor /code/vendor/
RUN echo "backend: Agg" > /code/matplotlibrc
ENV MATPLOTLIBRC=/code/matplotlibrc
