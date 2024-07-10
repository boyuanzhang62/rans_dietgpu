FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04 as dev-base

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git && \
    rm -rf /var/lib/apt/lists/*
ENV PATH /opt/conda/bin:$PATH

FROM dev-base as conda
ENV PYTHON_VER=3.9.5
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VER} pytorch pytorch-cuda=11.8 ninja cmake -c pytorch -c nvidia && \
    /opt/conda/bin/conda clean -ya

ENV CMAKE_PREFIX_PATH="/opt/conda/bin/../:/opt/conda/lib/python3.9/site-packages/torch/share/cmake"

CMD bash
