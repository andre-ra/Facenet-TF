FROM tensorflow/tensorflow:latest-gpu

ARG k_element
ARG dataset

ENV K_ELEMENT=${k_element}
ENV DATASET=${dataset}

RUN pip install progressbar2 matplotlib

COPY / /mydir

WORKDIR /Facenet-TF/kmeans_tests/${dataset}/${k_element}

CMD ["/bin/bash","../../../../mydir/train_loop_gpu.sh"]