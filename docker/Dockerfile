FROM tensorflow/tensorflow
RUN apt-get update && \
    apt-get --yes install inetutils-ping iproute2 git vim python3-tk graphviz && \
    git config --global --add safe.directory /deep-learning
RUN pip install --upgrade pip && \
    pip install ipython matplotlib pydot && \
    pip install torch && \
    pip install jax
COPY _bashrc /root/.bashrc
