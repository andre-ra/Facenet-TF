from cProfile import label
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

from src.params import Params
from src.model  import face_model
from src.data   import get_dataset
from src.triplet_loss import batch_all_triplet_loss, batch_hard_triplet_loss, adapted_triplet_loss, _pairwise_distances


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class Trainer():
    
    def __init__(self, json_path, data_dir, data_dir_valid, validate, ckpt_dir, log_dir, restore):
        
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
        
        if self.valid:
            self.valid_dataset, self.valid_samples = get_dataset(data_dir_valid, self.params, 'val')

        #self.train_dataset = tf.keras.utils.image_dataset_from_directory(
        #    data_dir,
        #    validation_split=0.25,
        #    subset="training",
        #    seed=123,
        #    image_size=(150,75),
        #    batch_size=108,
        #    label_mode='int',
        #    shuffle=True,
        #    color_mode ="grayscale"
        #)

        #self.train_samples = len(self.train_dataset)
        #train_ds = self.train_dataset.unbatch()
        #images = list(train_ds.map(lambda x, y: x))
        #labels = list(train_ds.map(lambda x, y: y))
        #print(len(images))
        #ll = []
        #for elem in labels:
        #    ll.append(elem.numpy())
        #print(ll)
        #print(len(ll))
        
    def __call__(self, epoch):
        
        for i in range(epoch):
            self.train(i)
            if self.valid:
                self.validate(i)

        
    def train(self, epoch):
        widgets = [f'Train epoch {epoch} :',progressbar.Percentage(), ' ', progressbar.Bar('#'), ' ',progressbar.Timer(), ' ', progressbar.ETA(), ' ']
        #print("aaa")
        #print(self.train_samples)
        #print(self.params.batch_size)
        pbar = progressbar.ProgressBar(widgets=widgets,max_value=int(self.train_samples / self.params.batch_size) + 10).start()
        total_loss = 0
        #shuffle_list = []
        #old_classes = []
        #new_classes = []
        #aaa = []
        #bbb= []
        #A = A.batch(18).prefetch(AUTOTUNE)
        #A = self.train_dataset.unbatch().batch(3)
        #for e in A:
        #    for i in range(0,len(e[1])):
        #        aaa.append(e[1][i].numpy())

        #A = A.shuffle(2000)
        #g = 0
        #for e in A:
        #    for i in range(0,len(e[1])):
        #        bbb.append(e[1][i].numpy())
        #        g = g + 1
        #print(g)
        #print("aaa")
        #print(aaa)
        #print("-------------------------------------------")
        #print(bbb)
        #for e in A:
        #    e[0][0] = tf.reshape(e[0],shape=(18,75,133,3))
        #print("ssss")
        #print(len(self.train_dataset))
        #print(len(A))
        #for e in A:
        #    last_image = None
        #    last_label = None
        #    for i in range(0,len(e[0])):
        #        if last_image is not None:
        #            print("ok")
        #            e[0][i] = tf.concat([e[0][i],last_image],0)
        #            tf.concat([e[1][i],last_label],0)
        #        last_image = e[0][i]
         #       last_label = e[1][i]
        #        print(last_image.shape)
        #    print(e[0].shape)
        #print(len(A))
        #for e in A:
        #    print(e[0].shape)

        #A = self.train_dataset.padded_batch(3)
        #for d in A:
        #    print(d)
        #for e in self.train_dataset:
        #    for i in range(0,len(e[0]),3):
        #        shuffle_list.append([ (e[0][i],e[1][i]), (e[0][i+1],e[1][i+1]), (e[0][i+2],e[1][i+2])])
                #print(e[0][i].shape)
                #print("aaaaaaaaaaa")
                #print(e[1][i].numpy())
                #print(i)
            #print(e[0].shape)
        #for val in shuffle_list:
        #    old_classes.append((val[0][1].numpy(),val[1][1].numpy(),val[2][1].numpy()))

        #print(old_classes)

        #print("---------------------------------------------------------------------------")

        #random.shuffle(shuffle_list)

        #for val in shuffle_list:
        #    new_classes.append((val[0][1].numpy(),val[1][1].numpy(),val[2][1].numpy()))
            #print(val[0][0].shape)
        #print(new_classes)
        b = 0
        #for e in self.train_dataset:
        #    for i in range(0,len(e[0]),3):
        #        shuffle_list.append([ (e[0][i],e[1][i]), (e[0][i+1],e[1][i+1]), (e[0][i+2],e[1][i+2])])
        #        e[0][i], e[1][i] = tf.variable(shuffle_list[b*i][0])
        #        tf.tensor(e[0][i], e[1][i])
        #        e[0][i+1], e[1][i+1] = tf.variable(shuffle_list[b*i][1])
        #        tf.tensor(e[0][i+1], e[1][i+1])
        #        e[0][i+2], e[1][i+2] = tf.variable(shuffle_list[b*i][2])
        #        tf.tensor(e[0][i+2], e[1][i+2])
            #print(e[0])
        #print(shuffle_list[0][0].shape)
        #dss = tf.data.Dataset.from_tensor_slices(shuffle_list[0][0][0])
        #print(shuffle_list[0][0][0])
        #print(dss)
        #print(self.params.batch_size)
        self.train_dataset = self.train_dataset.unbatch().batch(18)
        self.train_dataset = self.train_dataset.shuffle(1080)
        self.train_dataset = self.train_dataset.unbatch().batch(self.params.batch_size)
        #l=0
        #for e in self.train_dataset:
        #    l = l+1
        #for a in dss:
        #    print(dss)
        #print(l)
        for i, (images, labels) in pbar(enumerate(self.train_dataset)):
            loss = self.train_step(images, labels)
            total_loss += loss
            
            with self.train_summary_writer.as_default():
                tf.summary.scalar('train_step_loss', loss, step=self.checkpoint.train_steps)
            self.checkpoint.train_steps.assign_add(1)
        
        with self.train_summary_writer.as_default():
            tf.summary.scalar('train_batch_loss', total_loss, step=epoch)
        
        self.checkpoint.epoch.assign_add(1)
        if int(self.checkpoint.epoch) % 10 == 0:
            save_path = self.ckptmanager.save()
            print(f'Saved Checkpoint for step {self.checkpoint.epoch.numpy()} : {save_path}\n')
        print('\nTrain Loss over epoch {}: {}'.format(epoch, total_loss))

    def evaluate_train(self):
        train_embeddings = {}
        evaluate_train_embeddings = {}
        for train_images in self.train_dataset:
            #embedds = self.model(valid_images)
            #valid_embeddings.append((embedds,valid_labels[0].numpy()))
            train_embedds = self.model(train_images[0])
            #print(valid_embedds)
            for i in range(0,len(train_embedds)):
                #print(train_images[1][i].numpy())
                if train_images[1][i].numpy() not in train_embeddings:
                    train_embeddings[train_images[1][i].numpy()] = tf.expand_dims(train_embedds[i],axis=0)
                    evaluate_train_embeddings[train_images[1][i].numpy()] = [tf.expand_dims(train_embedds[i],axis=0)]
                else:
                    train_embeddings[train_images[1][i].numpy()] = tf.concat([train_embeddings[train_images[1][i].numpy()],tf.expand_dims(train_embedds[i],axis=0)],axis=0)
                    evaluate_train_embeddings[train_images[1][i].numpy()].append(tf.expand_dims(train_embedds[i],axis=0))
        print("t")
        print(len(train_embeddings))

        json_train_embeddings = {}
        for train_key, train_values in train_embeddings.items():
            for train_val in train_values:
                if str(train_key) not in json_train_embeddings:
                    print(str(train_key))
                    json_train_embeddings[str(train_key)] = [train_val.numpy().tolist()]
                else:
                    json_train_embeddings[str(train_key)].append(train_val.numpy().tolist())

        #print(json_train_embeddings["b'1'"])
        #ew_data = [dict(zip(json_train_embeddings.keys(), i)) for i in zip(*json_train_embeddings.values())]
        #print(new_data)
        json.dump(json_train_embeddings, codecs.open('train_dataset_embeddings.json', 'w', encoding='utf-8'), 
          separators=(',', ':'), 
          sort_keys=True, 
          indent=4)

        #with open('train_dataset_embeddings.json', 'w') as outfile:
        #    json.dump(json_train_embeddings, outfile)

        fails = 0
        totalops = 0
        nao_encontrados = []
        fails_list = []
        for valid_key, valid_values in evaluate_train_embeddings.items():
            minDist = 1000
            realDist = []
            for valls in valid_values:
                for train_key, train_values in train_embeddings.items():
                    concat_embedds = tf.concat([valls, train_values],axis=0)
                    distances = _pairwise_distances(concat_embedds,self.params.squared)
                    total_dist = tf.reduce_sum(distances[0])
                    if total_dist < minDist:
                        minimalDistances = distances[0]
                        minDist = total_dist
                        match_label = train_key
                    if train_key == valid_key:
                        realDistances = distances[0]
                        real_dist = total_dist
                if(match_label != valid_key):
                    fails_list.append(str(match_label) + " : " + str(minDist) + " , " + str(minimalDistances.numpy()) + " / " + str(valid_key) + " : " + str(real_dist) + " , " + str(realDistances.numpy()))
                    fails += 1
                totalops += 1
            if totalops%18 == 0:
                print(totalops)
                print("F " + str(fails))
        print("Quant fails")
        print(fails)
        print("Total ops")
        print(totalops)
        acc = (1-(fails/totalops))*100
        print("Accuracy : " + str(acc))
        print("Fails")
        print(fails_list)



    def evaluate(self):
        # Loads the weights
        #weightss = self.model.load_weights(self.ckptmanager.latest_checkpoint)

        valid_embeddings = {}
        for valid_images in self.valid_dataset:
            #print(valid_images[1][0].numpy())
            #embedds = self.model(valid_images)
            #valid_embeddings.append((embedds,valid_labels[0].numpy()))
            valid_embedds = self.model(valid_images[0])
            #print(valid_embedds)
            for i in range(0,len(valid_embedds)):
                if valid_images[1][i].numpy() not in valid_embeddings:
                    valid_embeddings[valid_images[1][i].numpy()] = [tf.expand_dims(valid_embedds[i],axis=0)]
                else:
                    #valid_embeddings[valid_images[1][i].numpy()] = tf.concat([valid_embeddings[valid_images[1][i].numpy()],tf.expand_dims(valid_embedds[i],axis=0)],axis=0)
                    valid_embeddings[valid_images[1][i].numpy()].append(tf.expand_dims(valid_embedds[i],axis=0))
                    #print("aaa")
                    #print(len(valid_embeddings[valid_images[1][i].numpy()]))
                #valid_embeddings.append([valid_embedds[i],valid_images[1][i]])
            #print(embeddings)
            #print(embeddings)
        print("b")
        print(len(valid_embeddings))
        #for k,v in valid_embeddings.items():
        #    print(k)

        train_embeddings = {}
        for train_images in self.train_dataset:
            #embedds = self.model(valid_images)
            #valid_embeddings.append((embedds,valid_labels[0].numpy()))
            train_embedds = self.model(train_images[0])
            #print(valid_embedds)
            for i in range(0,len(train_embedds)):
                #print(train_images[1][i].numpy())
                if train_images[1][i].numpy() not in train_embeddings:
                    train_embeddings[train_images[1][i].numpy()] = tf.expand_dims(train_embedds[i],axis=0)
                else:
                    train_embeddings[train_images[1][i].numpy()] = tf.concat([train_embeddings[train_images[1][i].numpy()],tf.expand_dims(train_embedds[i],axis=0)],axis=0)
                    #train_embeddings[train_images[1][i].numpy()] = train_embeddings[train_images[1][i].numpy()].append(tf.expand_dims(train_embedds[i],axis=0))
                #valid_embeddings.append([valid_embedds[i],valid_images[1][i]])
            #print(embeddings)
            #print(embeddings)
        print("t")
        print(len(train_embeddings))
        #print(train_embeddings.values())
            #valid_labels.append(labels)
            #for i in range(1, embedds.shape[0]):
            #    valid_embeddings[labels.numpy()[i]] = embedds.numpy()[i]
            #print(embedds)
        #for ell in valid_embeddings:
        #    print(ell[1])
        #print(valid_labels[0].numpy())
        fails = 0
        totalops = 0
        nao_encontrados = []
        fails_list = []
        #train_embeddings = []
        #for images in self.train_dataset:

            #for imgs in images:
            #    print(type(imgs))
        #for element in self.train_dataset:
            #print(element)
            #print(images.shape)
            #print(labels.numpy())
            #embeddings = self.model(images[0])
            #for i in range(0,len(embeddings)):
                #train_embeddings.append([embeddings[i],images[1][i]])
            #print(embeddings)
            #print(embeddings)
        #print("a")
        #print(len(train_embeddings))
        #print(train_embeddings[0])
        for valid_key, valid_values in valid_embeddings.items():
            minDist = 1000
            realDist = []
            #print(valid_values)
            for valls in valid_values:
                for train_key, train_values in train_embeddings.items():
                    concat_embedds = tf.concat([valls, train_values],axis=0)
                    distances = _pairwise_distances(concat_embedds,self.params.squared)
                    total_dist = tf.reduce_sum(distances[0])
                    if total_dist < minDist:
                        minimalDistances = distances[0]
                        minDist = total_dist
                        match_label = train_key
                    if train_key == valid_key:
                        realDistances = distances[0]
                        real_dist = total_dist
                if(match_label != valid_key):
                    fails_list.append(str(match_label) + " : " + str(minDist) + " , " + str(minimalDistances.numpy()) + " / " + str(valid_key) + " : " + str(real_dist) + " , " + str(realDistances.numpy()))
                    #fails_list.append(str(match_label) + " : " + str(minDist)  + " /  " + str(valid_key) + " : " + str(real_dist))
                    #print("Erro "+str(match_label))
                    #print("Certo "+str(valid_key))
                    #print(realDistances)
                    #print(real_dist)
                    #print(minimalDistances)
                    #print(minDist)
                    #print("dists")
                    fails += 1
                totalops += 1
            if totalops%18 == 0:
                print(totalops)
                print("F " + str(fails))

        #for vals in valid_embeddings:
        #    minDist = 1000
        #    realDist = []
        #    for trains in train_embeddings:
        #        concat_embedds = tf.concat([vals[0],trains[0]],0)
        #        distance = _pairwise_distances(concat_embedds,self.params.squared)
        #        if (distance[0][1] < minDist):
        #            minDist = distance[0][1]
        #            match_label = trains[1].numpy()
        #        if (trains[1].numpy() == vals[1].numpy()):
        #            realDist.append(distance[0][1])
        #    if (match_label != vals[1].numpy()):
        #        print("Erro "+str(match_label))
        #        print("Certo "+str(vals[1].numpy()))
        #        print(realDist)
          
          
            #    distances = _pairwise_distances(eval_embeddings,self.params.squared)
            #    dist = np.sum(distances[0].numpy())

                #if (dist < minDist):
                #    minDist = dist
                    #print(distances[0].numpy())
                    #print(dist)
                    #print(element[1])
                    #print(dist)
                #    fingerID = element[1]
                #if(fingerID == labels[0].numpy()):
                #    realDist = dist
                    #print("aaaaaaaaaa")
                    #print(distances[0].numpy())
                    #print(realDist)
                #print(type(labels.numpy()))
                #print(type(fingerID))
            #if (realDist == -1):
            #    nao_encontrados.append(str(labels[0].numpy()))
                #print("NAO ENCONTRADO"+ str(labels[0].numpy()))
            #elif (fingerID != labels[0].numpy()):
            #    fails_list.append(str(fingerID)+' '+str(minDist)+ ' : ' + str(labels[0].numpy()) + ' ' + str(realDist))
                #print(" Eval " + str(fingerID))
                #print("EvalDist " + str(minDist))
                #print("Train " + str(labels[0].numpy()))
                #print("RealDist "+ str(realDist))
                #fails +=1
            #totalops +=1
        print("Quant fails")
        print(fails)
        print("Total ops")
        print(totalops)
        acc = (1-(fails/totalops))*100
        print("Accuracy : " + str(acc))
        #print("Nao encontrados")
        #print(nao_encontrados)
        print("Fails")
        print(fails_list)


        #widgets = [f'bbbbbb :', progressbar.Percentage(), ' ', progressbar.Bar('#'), ' ',progressbar.Timer(), ' ', progressbar.ETA(), ' ']
        #pbar = progressbar.ProgressBar(widgets=widgets,max_value=int(self.valid_samples // self.params.batch_size) + 50).start()
        #valid_embeddings = []
        #valid_labels = []
        #for i, (valid_images, valid_labels) in pbar(enumerate(self.valid_dataset)):
        #    embedds = self.model(valid_images)
        #    valid_embeddings.append((embedds,valid_labels[0].numpy()))
            #valid_labels.append(labels)
            #for i in range(1, embedds.shape[0]):
            #    valid_embeddings[labels.numpy()[i]] = embedds.numpy()[i]
            #print(embedds)
        #for ell in valid_embeddings:
        #    print(ell[1])
        #print(valid_labels[0].numpy())
        #widgets = [f'aaaaaa:',progressbar.Percentage(), ' ', progressbar.Bar('#'), ' ',progressbar.Timer(), ' ', progressbar.ETA(), ' ']
        #pbar = progressbar.ProgressBar(widgets=widgets,max_value=int(self.train_samples // self.params.batch_size) + 20).start()
        #fails = 0
        #totalops = 0
        #nao_encontrados = []
        #fails_list = []
        #train_embeddings = []
        #for i, (images, labels) in pbar(enumerate(self.train_dataset)):

            #for imgs in images:
            #    print(type(imgs))
        #for element in self.train_dataset:
            #print(element)
            #print(images.shape)
            #print(labels.numpy())
            #embeddings = self.model(images)
            #for embeds in embeddings:
            #    train_embeddings.append((embeds,labels[0].numpy())
            #print(embeddings)
            #print(embeddings)
            #minDist = 1000
            #fingerID = ''
            #realDist = -1
            #for element in valid_embeddings:
                #elementList = [element]
                #embeddings.push(element)
                #dists = evaluate_distances(embeddings,element,self.params.squared)
            #    eval_embeddings = tf.concat([element[0],embeddings],0)
            #print(embeddings.shape)
            #    distances = _pairwise_distances(eval_embeddings,self.params.squared)
            #    dist = np.sum(distances[0].numpy())

            #    if (dist < minDist):
            #        minDist = dist
                    #print(distances[0].numpy())
                    #print(dist)
                    #print(element[1])
                    #print(dist)
            #        fingerID = element[1]
            #    if(fingerID == labels[0].numpy()):
            #        realDist = dist
                    #print("aaaaaaaaaa")
                    #print(distances[0].numpy())
                    #print(realDist)
                #print(type(labels.numpy()))
                #print(type(fingerID))
            #if (realDist == -1):
            #    nao_encontrados.append(str(labels[0].numpy()))
                #print("NAO ENCONTRADO"+ str(labels[0].numpy()))
            #elif (fingerID != labels[0].numpy()):
            #    fails_list.append(str(fingerID)+' '+str(minDist)+ ' : ' + str(labels[0].numpy()) + ' ' + str(realDist))
                #print(" Eval " + str(fingerID))
                #print("EvalDist " + str(minDist))
                #print("Train " + str(labels[0].numpy()))
                #print("RealDist "+ str(realDist))
            #    fails +=1
            #totalops +=1
        #print("Fails")
        #print(fails)
        #print("Total ops")
        #print(totalops)
        #print("Nao encontrados")
        #print(nao_encontrados)
        #print("Falhas")
        #print(fails_list)
            #print(minDist)
            #print(dists[0])
            #print(dists)
            #print(embeddings.shape)
            #for element in valid_embeddings:
            #embeddings = tf.concat([embeddings, embedds],0)
            #print(embeddings.shape)
            #distances = _pairwise_distances(embeddings,self.params.squared)
            #print(distances)
        #for j, (images, labels) in enumerate(self.train_dataset):
        #    embeddings = self.model(images)
            
            #embeddings = tf.math.l2_normalize(embeddings, axis=1, epsilon=1e-10)
            #distances = _pairwise_distances__(embeddings, squared=False, evaluate=True, evalImage=valid_images[i])
                
        # Re-evaluate the model
        #loss, acc = self.model.evaluate(self.valid_dataset, self.valid_samples, verbose=2)
        #sprint("Restored model, accuracy: {:5.2f}%".format(100 * acc))


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


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=0, type=int,
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
    args = parser.parse_args()
    #print(args.params_dir)
    
    trainer = Trainer(args.params_dir, args.data_dir, args.data_dir_valid, args.validate, args.ckpt_dir, args.log_dir, args.restore)
    
    for i in range(args.epoch):
        trainer.train(i)
    #trainer.evaluate()
    trainer.evaluate_train()
# 1 record - /root/shared_folder/Harish/Facenet/data
# 10 records - /root/shared_folder/Amaan/face/FaceNet-and-FaceLoss-collections-tensorflow2.0/data10faces_aligned_tfrcd
# Complete record - /root/shared_folder/Amaan/face/FaceNet-and-FaceLoss-collections-tensorflow2.0/data2/
# python train.py --params_dir ./hyperparameters/batch_fingervein.json --data_dir ./newUTFVP3/train --log_dir ./.logs/ --ckpt_dir ./.ckpt/ --restore 1
# python train.py --params_dir ./hyperparameters/batch_fingervein.json --data_dir ./newUTFVP3/train --log_dir ./.logs/ --ckpt_dir ./.ckpt/ --restore 1 --validate 1 --data_dir_valid ./newUTFVP3/validate



        for valid_key, valid_values in evaluate_train_embeddings.items():
            minDist = 1000
            for vals in valid_values:
                for train_key, train_values in evaluate_train_embeddings.items():
                    equal = 0
                    total_dist = 0
                    for train_vals in train_values:
                        concat_embedds = tf.concat([vals, train_vals],axis=0)
                        distances = _pairwise_distances(concat_embedds,self.params.squared)
                        total_dist += distances[0][1]
                        if distances[0][1] == 0:
                            equal = 1
                    dist_class = total_dist/(len(train_values) - equal)
                    if dist_class < minDist:
                        minDist = dist_class
                        match_label = train_key
                    if train_key == valid_key:
                        real_dist = total_dist
                if(match_label != valid_key):
                    fails_list.append( str(match_label) + " : " + str(minDist) + " / " + str(valid_key) + " : " + str(real_dist) )
                    fails += 1
                totalops += 1
            if totalops%18 == 0:
                print(totalops)
                print("F " + str(fails))