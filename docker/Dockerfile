FROM tensorflow/tensorflow
RUN apt-get update && \
    apt-get --yes install vim && \
    apt-get --yes install inetutils-ping && \
    apt-get --yes install iproute2 && \
    apt-get --yes install python3-tk
RUN pip install --upgrade pip && \
    pip install ipython && \
    pip install matplotlib

