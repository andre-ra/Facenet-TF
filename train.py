from cProfile import label
from genericpath import exists
from optparse import Values
import os
import argparse
import datetime
from random import shuffle
import tensorflow as tf
import progressbar
import numpy as np
import random
import json
import codecs
import time
import matplotlib.pyplot as plt
from collections import OrderedDict

from src.params import Params
from src.model  import face_model
from src.data   import get_dataset
from src.triplet_loss import batch_all_triplet_loss, batch_hard_triplet_loss, adapted_triplet_loss, _pairwise_distances


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class Trainer():
    
    def __init__(self, json_path, data_dir, data_dir_valid, validate, ckpt_dir, log_dir, restore):
        

        self.train_embeddings = {}

        self.params      = Params(json_path)
        self.valid       = 1 if validate == '1' else 0
        self.model       = face_model(self.params)
        
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.params.learning_rate,
                                                                          decay_steps=10000, decay_rate=0.96, staircase=True)
        self.optimizer   = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=0.1)
        
        self.checkpoint  = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer, train_steps=tf.Variable(0,dtype=tf.int64),
                                               valid_steps=tf.Variable(0,dtype=tf.int64), epoch=tf.Variable(0, dtype=tf.int64))
        self.ckptmanager = tf.train.CheckpointManager(self.checkpoint, ckpt_dir, 3)
        
        if self.params.triplet_strategy == "batch_all":
            self.loss = batch_all_triplet_loss
            
        elif self.params.triplet_strategy == "batch_hard":
            self.loss = batch_hard_triplet_loss
            
        elif self.params.triplet_strategy == "batch_adaptive":
            self.loss = adapted_triplet_loss
            
        current_time = datetime.datetime.now().strftime("%d-%m-%Y_%H%M%S")
        log_dir += current_time + '/train/'
        self.train_summary_writer = tf.summary.create_file_writer(log_dir)
            
        if restore == '1':
            self.checkpoint.restore(self.ckptmanager.latest_checkpoint)
            print(f'\nRestored from Checkpoint : {self.ckptmanager.latest_checkpoint}\n')
        
        else:
            print('\nIntializing from scratch\n')
            
        self.train_dataset, self.train_samples = get_dataset(data_dir, self.params, 'train')

        if(self.checkpoint.epoch.numpy()==0):
            train_accuracy = self.evaluate_train()
            train_accuracy_dict = {str(self.checkpoint.epoch.numpy()) : float(train_accuracy)}
            self.write_json(train_accuracy_dict,'train_accuracy.json')
        
        if self.valid:
            self.valid_dataset, self.valid_samples = get_dataset(data_dir_valid, self.params, 'val')

            if(self.checkpoint.epoch.numpy()==0):
                test_accuracy = self.evaluate()
                test_accuracy_dict = {str(self.checkpoint.epoch.numpy()) : float(test_accuracy)}
                self.write_json(test_accuracy_dict,'test_accuracy.json')
        
    def __call__(self, epoch):
        
        for i in range(epoch):
            self.train(i)
            if self.valid:
                self.validate(i)

        
    def train(self, epoch):

        widgets = [f'Train epoch {epoch} :',progressbar.Percentage(), ' ', progressbar.Bar('#'), ' ',progressbar.Timer(), ' ', progressbar.ETA(), ' ']
        pbar = progressbar.ProgressBar(widgets=widgets,max_value=int(self.train_samples / self.params.batch_size) + 10).start()
        total_loss = 0

        if tf.config.list_physical_devices('GPU'):
            # Returns a dict in the form {'current': <current mem usage>,
            #                             'peak': <peak mem usage>}
            json.dump(tf.config.experimental.get_memory_info('GPU:0'), codecs.open('gpu_memmory_usage.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

        begin = time.time()

        self.train_dataset = self.train_dataset.unbatch().batch(self.params.class_size)
        self.train_dataset = self.train_dataset.shuffle(2000)
        self.train_dataset = self.train_dataset.unbatch().batch(self.params.batch_size)

        for i, (images, labels) in pbar(enumerate(self.train_dataset)):
            loss = self.train_step(images, labels)
            total_loss += loss
            
            with self.train_summary_writer.as_default():
                tf.summary.scalar('train_step_loss', loss, step=self.checkpoint.train_steps)
            self.checkpoint.train_steps.assign_add(1)
        
        with self.train_summary_writer.as_default():
            tf.summary.scalar('train_batch_loss', total_loss, step=epoch)
        
        end = time.time()
        train_epoch_time = end - begin

        train_epoch_time_dict = {str(self.checkpoint.epoch.numpy()) : float(train_epoch_time)}
        self.write_json(train_epoch_time_dict,'train_epoch_time.json')

        total_loss_dict = {str(self.checkpoint.epoch.numpy()) : float(total_loss)}
        self.write_json(total_loss_dict,'total_loss.json')

        self.checkpoint.epoch.assign_add(1)
        if int(self.checkpoint.epoch) % 5 == 0:
            save_path = self.ckptmanager.save()
            print(f'Saved Checkpoint for step {self.checkpoint.epoch.numpy()} : {save_path}\n')


        if int(self.checkpoint.epoch) % 5 == 0:
            test_accuracy = self.evaluate()
            test_accuracy_dict = {str(self.checkpoint.epoch.numpy()) : float(test_accuracy)}
            self.write_json(test_accuracy_dict,'test_accuracy.json')
        
            train_accuracy = self.evaluate_train()
            train_accuracy_dict = {str(self.checkpoint.epoch.numpy()) : float(train_accuracy)}
            self.write_json(train_accuracy_dict,'train_accuracy.json')

        print('\nTrain Loss over epoch {}: {}'.format(epoch, total_loss))


    def write_json(self, new_data, filename):
        if exists(filename):

            with open(filename, "r+") as file:
                data = json.load(file)
                data.update(new_data)
                file.seek(0)
                json.dump(data, file, separators=(',', ':'), sort_keys=True, indent=4)
        else:
            json.dump(new_data, codecs.open(filename, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

    def evaluate_train(self):
        
        begin = time.time()
        train_embeddings = {}
        evaluate_train_embeddings = {}
        for train_images in self.train_dataset:
            train_embedds = self.model(train_images[0])

            for i in range(0,len(train_embedds)):
                if train_images[1][i].numpy() not in train_embeddings:
                    train_embeddings[train_images[1][i].numpy()] = tf.expand_dims(train_embedds[i],axis=0)
                    evaluate_train_embeddings[train_images[1][i].numpy()] = [tf.expand_dims(train_embedds[i],axis=0)]
                else:
                    train_embeddings[train_images[1][i].numpy()] = tf.concat([train_embeddings[train_images[1][i].numpy()],tf.expand_dims(train_embedds[i],axis=0)],axis=0)
                    evaluate_train_embeddings[train_images[1][i].numpy()].append(tf.expand_dims(train_embedds[i],axis=0))

        self.train_embeddings = train_embeddings

        json_train_embeddings = {}
        for train_key, train_values in train_embeddings.items():
            for train_val in train_values:
                if str(train_key) not in json_train_embeddings:
                    json_train_embeddings[str(train_key)] = [train_val.numpy().tolist()]
                else:
                    json_train_embeddings[str(train_key)].append(train_val.numpy().tolist())

        json.dump(json_train_embeddings, codecs.open('train_dataset_embeddings.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

        fails = 0
        totalops = 0

        for valid_key, valid_values in evaluate_train_embeddings.items():
            minDist = 1000
            for valls in valid_values:
                for train_key, train_values in train_embeddings.items():
                    concat_embedds = tf.concat([valls, train_values],axis=0)
                    distances = _pairwise_distances(concat_embedds,self.params.squared)
                    total_dist = tf.reduce_sum(distances[0])

                    if train_key == valid_key:
                        total_dist = total_dist/(len(train_values)-1)
                    else:
                        total_dist = total_dist/(len(train_values))
                    if total_dist < minDist:
                        minDist = total_dist
                        match_label = train_key

                if(match_label != valid_key):
                    fails += 1
                totalops += 1

        print("Quant fails")
        print(fails)
        print("Quant comps")
        print(totalops)
        acc = (1-(fails/totalops))*100
        print("Accuracy : " + str(acc))

        end = time.time()
        evaluate_train_epoch_time = end - begin
        print("Time " + str(evaluate_train_epoch_time))

        evaluate_train_epoch_time_dict = {str(self.checkpoint.epoch.numpy()) : float(evaluate_train_epoch_time)}
        self.write_json(evaluate_train_epoch_time_dict,'evaluate_train_epoch_time.json')
        train_accuracy_dict = {str(self.checkpoint.epoch.numpy()) : float(acc)}
        self.write_json(train_accuracy_dict,'train_accuracy.json')

        return acc



    def evaluate(self):

        begin = time.time()
        valid_embeddings = {}
        for valid_images in self.valid_dataset:
            valid_embedds = self.model(valid_images[0])

            for i in range(0,len(valid_embedds)):
                if valid_images[1][i].numpy() not in valid_embeddings:
                    valid_embeddings[valid_images[1][i].numpy()] = [tf.expand_dims(valid_embedds[i],axis=0)]
                else:
                    valid_embeddings[valid_images[1][i].numpy()].append(tf.expand_dims(valid_embedds[i],axis=0))

        if self.train_embeddings:
            train_embeddings = self.train_embeddings
        else:
            train_embeddings = {}
            for train_images in self.train_dataset:
                train_embedds = self.model(train_images[0])

                for i in range(0,len(train_embedds)):
                    if train_images[1][i].numpy() not in train_embeddings:
                        train_embeddings[train_images[1][i].numpy()] = tf.expand_dims(train_embedds[i],axis=0)
                    else:
                        train_embeddings[train_images[1][i].numpy()] = tf.concat([train_embeddings[train_images[1][i].numpy()],tf.expand_dims(train_embedds[i],axis=0)],axis=0)

        fails = 0
        totalops = 0

        for valid_key, valid_values in valid_embeddings.items():
            minDist = 1000

            for valls in valid_values:
                for train_key, train_values in train_embeddings.items():
                    concat_embedds = tf.concat([valls, train_values],axis=0)
                    distances = _pairwise_distances(concat_embedds,self.params.squared)
                    total_dist = tf.reduce_sum(distances[0])/len(train_values)

                    if total_dist < minDist:
                        minDist = total_dist
                        match_label = train_key
                if(match_label != valid_key):
                    fails += 1
                totalops += 1

        print("Quant fails")
        print(fails)
        print("Quant comps")
        print(totalops)
        acc = (1-(fails/totalops))*100
        print("Accuracy : " + str(acc))

        end = time.time()
        evaluate_epoch_time = end - begin
        print("Time " + str(evaluate_epoch_time))

        evaluate_epoch_time_dict = {str(self.checkpoint.epoch.numpy()) : float(evaluate_epoch_time)}
        self.write_json(evaluate_epoch_time_dict,'evaluate_epoch_time.json')

        test_accuracy_dict = {str(self.checkpoint.epoch.numpy()) : float(acc)}
        self.write_json(test_accuracy_dict,'test_accuracy.json')

        return acc


    def validate(self, epoch):
        widgets = [f'Valid epoch {epoch} :', progressbar.Percentage(), ' ', progressbar.Bar('#'), ' ',progressbar.Timer(), ' ', progressbar.ETA(), ' ']
        pbar = progressbar.ProgressBar(widgets=widgets,max_value=int(self.valid_samples // self.params.batch_size) + 50).start()
        total_loss = 0

        for i, (images, labels) in pbar(enumerate(self.valid_dataset)):
            loss = self.valid_step(images, labels)
            total_loss += loss
            
            with self.train_summary_writer.as_default():
                tf.summary.scalar('valid_step_loss', loss, step=self.checkpoint.valid_steps)
            self.checkpoint.valid_steps.assign_add(1)
        print('\n')
        with self.train_summary_writer.as_default():
            tf.summary.scalar('valid_batch_loss', total_loss, step=epoch)
        
        if (epoch+1)%5 == 0:
            print('\nValidation Loss over epoch {}: {}\n'.format(epoch, total_loss)) 
    
        
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            embeddings = self.model(images)
            embeddings = tf.math.l2_normalize(embeddings, axis=1, epsilon=1e-10)
            loss = self.loss(labels, embeddings, self.params.margin, self.params.squared)
            
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss
    
    
    def valid_step(self, images, labels):
        
        embeddings = self.model(images)
        embeddings = tf.math.l2_normalize(embeddings, axis=1, epsilon=1e-10)
        loss = self.loss(labels, embeddings, self.params.margin, self.params.squared)
            
        return loss

    def predict_sample(self,sample):

        begin = time.time()

        f = open('train_dataset_embeddings.json')
        train_dict_loaded = json.load(f)

        minDist = 1000
        image_string = tf.io.read_file(sample)
        image = tf.image.decode_png(image_string, channels=3)
        image = tf.image.resize(image, [170, 306])
        image = image/ 255

        sample_embedd = self.model(tf.expand_dims(image,axis=0))
        
        for train_key, train_values in train_dict_loaded.items():
            train_values = tf.convert_to_tensor(train_values, dtype=tf.float32)
            concat_embedds = tf.concat([sample_embedd, train_values],axis=0)
            distances = _pairwise_distances(concat_embedds,self.params.squared)
            total_dist = tf.reduce_sum(distances[0])

            if total_dist < minDist:
                minDist = total_dist
                match_label = train_key

        end = time.time()
        return match_label, end - begin

def plot(data_file_list, save_file, title, y_axis_label, x_axis_label = 'Épocas', quant_cut_elements=0):

    plt.figure()
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(title)

    for data_file, data_label in data_file_list:
        f = open(data_file)
        data_dict = json.load(f)
        data_dict = {int(k):v for k,v in data_dict.items()}
        data_dict = OrderedDict(sorted(data_dict.items()))
        x_axis = list(data_dict.keys())
        y_axis = list(data_dict.values())
        x_axis = x_axis[quant_cut_elements:len(x_axis)]
        y_axis = y_axis[quant_cut_elements:len(y_axis)]
        plt.plot(x_axis,y_axis, label=data_label)

    plt.legend(loc='best')
    plt.savefig(save_file)




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=5, type=int,
                        help="Number epochs to train the model for")
    parser.add_argument('--params_dir', default='hyperparameters/batch_adaptive.json',
                        help="Experiment directory containing params.json")
    parser.add_argument('--data_dir', default='../face-data/',
                        help="Directory containing the dataset")
    parser.add_argument('--data_dir_valid', default='../face-data/validate',
                        help="Directory containing the dataset")
    parser.add_argument('--validate', default='0',
                        help="Is there an validation dataset available")
    parser.add_argument('--ckpt_dir', default='.tf_ckpt/',
                        help="Directory containing the Checkpoints")
    parser.add_argument('--log_dir', default='.logs/',
                        help="Directory containing the Logs")
    parser.add_argument('--restore', default='0',
                        help="Restart the model from the previous Checkpoint")
    parser.add_argument('--predict', default='0',
                        help="Given an image, predicts from which class it belongs")
    parser.add_argument('--predict_sample', default='../face-data/validate/1.png',
                        help="Image that will be predicted the class")
    parser.add_argument('--plot', default='0',
                        help="Plot graphics of the dataset")
    args = parser.parse_args()

    if args.plot == '1':
        plot([('FullPreProcessing/test_accuracy.json',1),('Raw/test_accuracy.json',2),('FullPreProcessing360/test_accuracy.json',3)],'graficos/test_accuracy_graphic.png', 'Acurácia por época no dataset de teste', 'Acurácia', 'Épocas',0)
        plot([('FullPreProcessing/test_accuracy.json',1),('Raw/test_accuracy.json',2),('FullPreProcessing360/test_accuracy.json',3)],'graficos/test_accuracy_graphic10.png', 'Acurácia por época no dataset de teste', 'Acurácia', 'Épocas',9)
        plot([('FullPreProcessing/train_accuracy.json',1),('Raw/train_accuracy.json',2),('FullPreProcessing360/train_accuracy.json',3)],'graficos/train_accuracy_graphic.png', 'Acurácia por época no dataset de treino', 'Acurácia', 'Épocas',0)
        plot([('FullPreProcessing/train_accuracy.json',1),('Raw/train_accuracy.json',2),('FullPreProcessing360/train_accuracy.json',3)],'graficos/train_accuracy_graphic10.png', 'Acurácia por época no dataset de treino', 'Acurácia', 'Épocas',9)
        plot([('FullPreProcessing/total_loss.json', 1),('Raw/total_loss.json',2),('FullPreProcessing360/total_loss.json',3)],'graficos/total_loss_graphic.png', 'Perda(custo) por época', 'Perda(custo)', 'Épocas',0)
        plot([('FullPreProcessing/total_loss.json', 1),('Raw/total_loss.json',2),('FullPreProcessing360/total_loss.json',3)],'graficos/total_loss_graphic50.png', 'Perda(custo) por época', 'Perda(custo)', 'Épocas',49)
    else:
        trainer = Trainer(args.params_dir, args.data_dir, args.data_dir_valid, args.validate, args.ckpt_dir, args.log_dir, args.restore)

        if args.predict == '1':
            pred_class, pred_time = trainer.predict_sample(args.predict_sample)
            print("Classe da imagem inserida " + str(pred_class))
            print("Tempo para realizar a previsão da classe " + str(pred_time))

        elif args.validate == '1':
                trainer.evaluate_train()
                trainer.evaluate()
        else:
            for i in range(args.epoch):
                trainer.train(i)


# 1 record - /root/shared_folder/Harish/Facenet/data
# 10 records - /root/shared_folder/Amaan/face/FaceNet-and-FaceLoss-collections-tensorflow2.0/data10faces_aligned_tfrcd
# Complete record - /root/shared_folder/Amaan/face/FaceNet-and-FaceLoss-collections-tensorflow2.0/data2/