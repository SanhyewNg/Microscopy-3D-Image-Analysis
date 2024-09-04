FROM python:3.6-stretch
WORKDIR /code
RUN echo "deb http://deb.debian.org/debian stretch-backports main" >> /etc/apt/sources.list
RUN echo "backend: Agg" >> matplotlibrc
RUN apt-get update && apt-get install -y -t stretch-backports openjdk-8-jdk gnuplot
ADD requirements.txt .
RUN sed -i -- "s|tensorflow-gpu|tensorflow|g" requirements.txt
RUN pip install -r requirements.txt
ADD . /code
