FROM ubuntu:16.04
MAINTAINER ishiguro

ENV HOME  /
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
ENV PYTHONIOENCODING "utf-8"

COPY . /app/

RUN rm -rf /var/lib/apt/lists/*
RUN apt-get update -y
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install \
  build-essential \
  curl \
  git \
  libbz2-dev \
  libfontconfig1 \
  libglib2.0-0 \
  libsm6 \
  libssl-dev \
  libreadline-dev \
  libsqlite3-dev \
  libxext6 \
  libxrender1 \
  locales \
  make \
  python-dev \
  python-pip \
  sudo \
  tzdata \
  vim \
  zlib1g-dev

RUN echo "${TZ}" > /etc/timezone && \
  rm /etc/localtime && \
  ln -s /usr/share/zoneinfo/Asia/Tokyo /etc/localtime && \
  dpkg-reconfigure -f noninteractive tzdata

RUN git clone https://github.com/yyuu/pyenv.git .pyenv
RUN pyenv install 3.6.3
RUN pyenv global 3.6.3
RUN sudo locale-gen en_US.UTF-8

WORKDIR app
RUN pip install -r requirement.txt
