# Deep Learning
This docker container was created to use it with the book:
Deep Learning with Python, Third Edition
https://www.manning.com/books/deep-learning-with-python-third-edition

## Use matplotlib within docker container
### Enable X11 display

```
$ xhost + 
access control disabled, clients can connect from any host
```
This allows to access the X11 server. This configuration is not save, because it allows the access from any host.
### Run docker container

```bash
docker run -it --name dl --net=host --env="DISPLAY" --volume="$HOME/.Xauthority:/root/.Xauthority:rw" blackhypothesis/dl:latest bash
```


