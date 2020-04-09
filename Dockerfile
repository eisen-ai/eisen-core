FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

RUN pip install --upgrade git+https://github.com/eisen-ai/eisen-core.git
RUN pip install --upgrade git+https://github.com/eisen-ai/eisen-cli.git