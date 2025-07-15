FROM tensorflow/tensorflow
RUN apt-get update && \
    apt-get --yes install inetutils-ping iproute2 git vim python3-tk && \
    git config --global --add safe.directory /deep-learning
RUN pip install --upgrade pip && \
    pip install ipython matplotlib torch
COPY _bashrc /root/.bashrc
