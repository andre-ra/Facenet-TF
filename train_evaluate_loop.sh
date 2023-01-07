#!/bin/bash

for i in {1..2}
do
    echo "Welcome $i times"
    if [ ! "$(docker ps -a | grep mycontainergpu)" ] 
    then
        echo "aaaaaaaaaaa"
        #sudo docker run -v /mnt/e/andre/Documents/FaculTeste/COURSERA/TensorFlow2-coursera/my_jupyter_andrewng/Facenet-TF:/Facenet-TF --name mycontainercpu andreriesco/tensorflow-cpu
        sudo docker run --gpus all -v /mnt/e/andre/Documents/FaculTeste/COURSERA/TensorFlow2-coursera/my_jupyter_andrewng/Facenet-TF:/Facenet-TF --name mycontainergpu andreriesco/tensorflow-gpu
    else
        #sudo docker start -i mycontainercpu
        sudo docker start -i mycontainergpu
    fi
done