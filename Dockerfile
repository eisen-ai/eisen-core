FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

RUN apt-get update
RUN apt-get install -y git

RUN pip install --upgrade git+https://github.com/eisen-ai/eisen-core.git
RUN pip install --upgrade git+https://github.com/eisen-ai/eisen-cli.git