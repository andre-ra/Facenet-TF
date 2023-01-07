for i in {1..1}
do
   echo "Welcome $i times"
   python ../../train.py --params_dir ../../hyperparameters/batch_fingervein.json --data_dir ./newUTFVPFullPreProcessing360/train --log_dir ./.logs/ --ckpt_dir ./.ckpt2/ --restore 1 --validate 0
done