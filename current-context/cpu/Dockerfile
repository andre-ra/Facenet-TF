FROM tensorflow/tensorflow:latest

RUN pip install progressbar2 matplotlib

COPY / /mydir

WORKDIR /Facenet-TF/UTFVPTests-CPU/CLAHE

CMD ["/bin/bash","../../../mydir/train_loop_cpu.sh"]