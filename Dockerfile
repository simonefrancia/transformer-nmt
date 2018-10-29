
FROM ubuntu:16.04
MAINTAINER Alberto Soragna alberto dot soragna @gmail.com

# working directory
ENV HOME /root
WORKDIR $HOME

RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    vim \
    nano \
    python-pip python-dev

# pip
RUN pip install --upgrade pip
    

#RUN export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}/usr/local/cuda/extras/CUPTI/lib64
RUN \
    pip install tensorflow && \
    pip install tensor2tensor

COPY \
    . $HOME/tensorflow-nlp

RUN \
    cd tensorflow-nlp && \
    pip install -r script/requirements.txt && \
    pip install regex && \
    pip install nltk && \
    cd - && \
    pip install jupyter

CMD jupyter notebook --no-browser --ip 0.0.0.0 --port $PORT  $HOME/developer_tensorflow-nlp


EXPOSE $PORT

WORKDIR $HOME/developer_tensorflow-nlp
