
python ../../../train.py --params_dir ../batch_fingervein.json --data_dir ../$DATASET --log_dir ./.logs/ --ckpt_dir ./.ckpt1/ --restore 1 --validate 1 --k_element $K_ELEMENT
