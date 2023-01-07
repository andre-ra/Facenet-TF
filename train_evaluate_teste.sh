#!/bin/bash

datasets=("UTFVPPP")

for ds in ${!datasets[@]}
do
    for j in {4..4}
    do
        sudo docker rm $(docker ps -a -q)
        docker rmi  andreriesco/tensorflow-gpu:latest
        cd current-context/gpu/
        sudo docker build --build-arg k_element=$j --build-arg dataset=${datasets[ds]} . -t andreriesco/tensorflow-gpu:latest
        cd ../..

        for i in {1..10}
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
    done
done