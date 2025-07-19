# Deep Learning
This docker container was created to use it with the book: \
__Deep Learning with Python, Third Edition__ \
https://www.manning.com/books/deep-learning-with-python-third-edition

This configuration is just to fulfill the requirements to run the code in the book. It does not take care about security related topics.

## Use matplotlib within docker container
### Enable X11 display

```
$ xhost + 
access control disabled, clients can connect from any host
```
This allows to access the X11 server. 
### Run docker container
Navigate to the directory, where this git repository was cloned.
```bash
docker run -it --name dl --net=host --env="DISPLAY" --user $(id -u):$(id -g) \
    --volume="$HOME/.Xauthority:/root/.Xauthority:rw" --volume="$(pwd):/deep-learning:rw" \
    --name deep_learning blackhypothesis/deep-learning:latest bash 
```
Now you can access the code within the docker container in the directory `/deep-learning`.

### Test 
Run the __test.py__. If it shows a 28 x 28 pixel image of the number 9, then it works.
```
cd /deep-learning
ipython
run test.py
```

## Visual Studio Code
### Remote Access To Docker Image
* Install the extension [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).
* Press [F1] and search for "Dev Containers: Attatch to Running Container ..."
* Choose the previously started container __  deep_learning__.
