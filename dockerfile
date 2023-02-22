# docker build -t dleongsh/torchaudio:1.13.6-cuda11.3-cudnn8-runtime .

#use the base package for from official pytorch source
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install libsndfile1 (linux soundfile package)
RUN apt-get update && apt-get install -y build-essential libsndfile1 git ffmpeg sox wget \
&& rm -rf /var/lib/apt/lists/*

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

# Install pip requirements
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

#docker container starts with bash
WORKDIR /
RUN ["bash"]