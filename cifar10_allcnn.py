# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 13:13:53 2016

@author: jeandut
"""

import numpy as np
import tensorflow as tf
import os
import cPickle as pickle
import argparse
import dataset_class as dataset
#
parser=argparse.ArgumentParser(description="Testing All-CNN on CIFAR-10 global contrast normalised and whitened without data-augmentation.")
parser.add_argument('--learning_rate', default=0.05, help="Initial Learning Rate")
parser.add_argument('--weight_decay',default=0.001, help="weight decay")
parser.add_argument('--data_dir',default="/home/mordan/Bureau", help="Directory with cifar-10-batches-py")

args=parser.parse_args()

WD=float(args.weight_decay)
starter_learning_rate=float(args.learning_rate)
batchdir=os.path.join(args.data_dir,"cifar-10-batches-py")

workdir=os.getcwd()

BATCH_SIZE=128
EPSILON_SAFE_LOG=np.exp(-50.)

#Loading CIFAR-10 from data_dir directory

train_batches=[os.path.join(batchdir, "data_batch_"+str(i)) for i in range(1,6)]

Xlist, ylist=[], []
for batch in train_batches:
    with open(batch,'rb') as f:
        d=pickle.load(f)
        Xlist.append(d['data'])
        ylist.append(d['labels'])
        
X_train=np.vstack(Xlist)
y_train=np.vstack(ylist)

with open(os.path.join(batchdir,"test_batch"),'rb') as f:
    d=pickle.load(f)
    X_test, y_test= d['data'], d['labels']
    
y_train=np.reshape(y_train,(-1,1))
y_test=np.array(y_test).reshape(-1, 1)



#Applying gcn followed by whitening

def global_contrast_normalize(X, scale=1., min_divisor=1e-8):
    
    X=X-X.mean(axis=1)[:, np.newaxis]
    
    normalizers=np.sqrt((X**2).sum(axis=1))/ scale
    normalizers[normalizers < min_divisor]= 1.
    
    X /= normalizers[:, np.newaxis]
    
    return X
    
def compute_zca_transform(imgs, filter_bias=0.1):
    
    meanX=np.mean(imgs,0)
    
    covX=np.cov(imgs.T)
    
    D, E =np.linalg.eigh(covX+ filter_bias * np.eye(covX.shape[0], covX.shape[1]))
    
    assert not np.isnan(D).any()
    assert not np.isnan(E).any()
    assert D.min() > 0
    
    D= D** -0.5
    
    W=np.dot(E, np.dot(np.diag(D), E.T))
    return meanX, W

def zca_whiten(train, test, cache=None):
    if cache and os.path.isfile(cache):
        with open(cache,'rb') as f:
            (meanX, W)=pickle.load(f)
    else:
        meanX, W=compute_zca_transform(train)
        
        with open(cache,'wb') as f:
            pickle.dump((meanX,W), f , 2)
            
    train_w=np.dot(train-meanX, W)
    test_w=np.dot(test-meanX, W)
    
    return train_w, test_w
    

norm_scale=55.0
X_train=global_contrast_normalize(X_train, scale= norm_scale)
X_test=global_contrast_normalize(X_test, scale= norm_scale)

zca_cache=os.path.join(workdir,'cifar-10-zca-cache.pkl')
X_train, X_test=zca_whiten(X_train, X_test, cache=zca_cache)


#Reformatting data as images
X_train=X_train.reshape((X_train.shape[0],3,32,32)).transpose((0,2,3,1))
X_test=X_test.reshape((X_test.shape[0],3,32,32)).transpose((0,2,3,1))

#Reformatting labels with 16 one-hot encoding
one_hot_train=np.zeros((y_train.shape[0],16),dtype="int64")
one_hot_test=np.zeros((y_test.shape[0],16),dtype="int64")

for i in xrange(y_train.shape[0]):
    one_hot_train[i,y_train[i]]=1
for i in xrange(y_test.shape[0]):
    one_hot_test[i,y_test[i]]=1
    
y_train=one_hot_train.astype("float32")
y_test=one_hot_test.astype("float32")

CIFAR10=dataset.read_data_sets(X_train,y_train, X_test, y_test, None, False)
 
def _variable_with_weight_decay(shape,wd=WD):
    
    initial=tf.random_normal(shape,stddev=0.05)
    
    var=tf.Variable(initial)
    
    if wd is not None:
        weight_decay=tf.mul(tf.nn.l2_loss(var), wd, name="weight_loss")
        tf.add_to_collection("losses", weight_decay)
        
    return var

    
def conv(input_tensor,W):
    
    return tf.nn.conv2d(input_tensor, W, strides=[1, 1, 1, 1], padding="VALID")
    
def convp1(input_tensor,W):
    
    return tf.nn.conv2d(input_tensor, W, strides=[1, 1, 1, 1], padding="SAME")
    
def convp1s2(input_tensor,W):
    
    return tf.nn.conv2d(input_tensor, W, strides=[1, 2, 2, 1], padding="SAME")
    
def avg_pool(input_tensor):
    return tf.nn.avg_pool(input_tensor, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding="VALID")
    
def safelog(input_tensor):
    return tf.log(input_tensor+EPSILON_SAFE_LOG)
    
    

x=tf.placeholder(dtype=tf.float32, shape= [None, 32, 32, 3])

#Padding to 16 classes to match Neon implementation
y_=tf.placeholder(dtype=tf.float32, shape= [None, 16])

#Placeholders for the dropout probabilities
keep_prob_input=tf.placeholder(tf.float32)
keep_prob_layers=tf.placeholder(tf.float32)

global_step=tf.Variable(0, trainable=False)


#Scheduling learning rate to drop from starter_learning_rate by a factor 10 after 200, 250 and 300 epochs with Momentum optimizer with momentum=0.9
def train(total_loss, global_step):
    
    NUM_EPOCHS_PER_DECAY_1=200
    NUM_EPOCHS_PER_DECAY_2=250
    NUM_EPOCHS_PER_DECAY_3=300
    
    LEARNING_RATE_DECAY_FACTOR=0.1
    num_batches_per_epoch=50000/128
    
    decay_steps_1=int(num_batches_per_epoch*NUM_EPOCHS_PER_DECAY_1)
    decay_steps_2=int(num_batches_per_epoch*NUM_EPOCHS_PER_DECAY_2)
    decay_steps_3=int(num_batches_per_epoch*NUM_EPOCHS_PER_DECAY_3)
    
    
    
    
    
    decayed_learning_rate_1=tf.train.exponential_decay(starter_learning_rate, 
                                                     global_step, 
                                                     decay_steps_1, 
                                                     LEARNING_RATE_DECAY_FACTOR,
                                                     staircase=True)
                                                     
    decayed_learning_rate_2=tf.train.exponential_decay(decayed_learning_rate_1, 
                                                     global_step, 
                                                     decay_steps_2, 
                                                     LEARNING_RATE_DECAY_FACTOR,
                                                     staircase=True)
                                                     
    decayed_learning_rate_3=tf.train.exponential_decay(decayed_learning_rate_2, 
                                                     global_step, 
                                                     decay_steps_3, 
                                                     LEARNING_RATE_DECAY_FACTOR,
                                                     staircase=True)
                                                     
    lr=decayed_learning_rate_3
    
    return tf.train.MomentumOptimizer(lr,0.9).minimize(total_loss,global_step) 


#Building Network

def inference(x_input):

    
    x_dropped=tf.nn.dropout(x_input, keep_prob=keep_prob_input)
    
    W_conv1=_variable_with_weight_decay([3,3,3,96])
    b_conv1=_variable_with_weight_decay([96])
    
    conv_int=convp1(x_dropped, W_conv1)
    biases=tf.nn.bias_add(conv_int, b_conv1)
    conv1=tf.nn.relu(biases)
    
    W_conv2=_variable_with_weight_decay([3,3,96,96])
    b_conv2=_variable_with_weight_decay([96])
    
    conv_int=convp1(conv1, W_conv2)
    biases=tf.nn.bias_add(conv_int, b_conv2)
    conv2=tf.nn.relu(biases)
    
    W_conv3=_variable_with_weight_decay([3,3,96,96])
    b_conv3=_variable_with_weight_decay([96])
    
    conv_int=convp1s2(conv2, W_conv3)
    biases=tf.nn.bias_add(conv_int, b_conv3)
    conv3=tf.nn.relu(biases)
    
    conv3_dropped=tf.nn.dropout(conv3, keep_prob=keep_prob_layers)
    
    W_conv4=_variable_with_weight_decay([3,3,96,192])
    b_conv4=_variable_with_weight_decay([192])
    
    conv_int=convp1(conv3_dropped, W_conv4)
    biases=tf.nn.bias_add(conv_int, b_conv4)
    conv4=tf.nn.relu(biases)
    
    W_conv5=_variable_with_weight_decay([3,3,192,192])
    b_conv5=_variable_with_weight_decay([192])
    
    conv_int=convp1(conv4, W_conv5)
    biases=tf.nn.bias_add(conv_int, b_conv5)
    conv5=tf.nn.relu(biases)
    
    W_conv6=_variable_with_weight_decay([3,3,192,192])
    b_conv6=_variable_with_weight_decay([192])
    
    conv_int=convp1s2(conv5, W_conv6)
    biases=tf.nn.bias_add(conv_int, b_conv6)
    conv6=tf.nn.relu(biases)
    
    conv6_dropped=tf.nn.dropout(conv6, keep_prob=keep_prob_layers)
    
    W_conv7=_variable_with_weight_decay([3,3,192,192])
    b_conv7=_variable_with_weight_decay([192])
    
    conv_int=convp1(conv6_dropped, W_conv7)
    biases=tf.nn.bias_add(conv_int, b_conv7)
    conv7=tf.nn.relu(biases)
    
    W_conv8=_variable_with_weight_decay([1,1,192,192])
    b_conv8=_variable_with_weight_decay([192])
    
    conv_int=conv(conv7, W_conv8)
    biases=tf.nn.bias_add(conv_int, b_conv8)
    conv8=tf.nn.relu(biases)
    
    W_conv9=_variable_with_weight_decay([1,1,192,16])
    b_conv9=_variable_with_weight_decay([16])
    
    conv_int=conv(conv8, W_conv9)
    biases=tf.nn.bias_add(conv_int, b_conv9)
    conv9=tf.nn.relu(biases)
    
    logits=avg_pool(conv9)
    
    return logits

    
def ce(y_pred, labels):
    cross_entropy=tf.reduce_sum(labels*safelog(y_pred),1)
    cross_entropy_mean=-tf.reduce_mean(cross_entropy)
    tf.add_to_collection("losses", cross_entropy_mean)
    
    return tf.add_n(tf.get_collection("losses"),"total_loss")
    
def acc(y_pred, labels):
    correct_prediction =tf.equal(tf.argmax(y_pred,1), tf.argmax(labels,1))
    return tf.reduce_mean(tf.cast(correct_prediction,tf.float32),0)



logits=inference(x)

pred=tf.nn.softmax(tf.reshape(logits,(-1,16)))

loss=ce(pred, y_)

train_step=train(loss, global_step)

accuracy=acc(pred, y_)

saver=tf.train.Saver()

sess=tf.Session()
sess.run(tf.initialize_all_variables())

#Going through 350 epochs (there is 390 batches by epoch)
STEPS, ACC_TRAIN, COST_TRAIN=[], [], []

for i in xrange(0,136500):
    batch=CIFAR10.train.next_batch(128)
    if i%100==0:
        acc_batch, loss_batch = sess.run([accuracy, loss], feed_dict={x: batch[0], y_: batch[1], keep_prob_input: 1., keep_prob_layers: 1.})
        print "Step: %s, Acc: %s, Loss: %s"%(i,acc_batch, loss_batch)
        
        STEPS.append(i)
        ACC_TRAIN.append(acc_batch)
        COST_TRAIN.append(loss_batch)
        
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob_input: 0.8, keep_prob_layers: 0.5})
    
save_path= saver.save(sess,"../temp/CIFAR10_allcnn")   
print "Model saved in file: ", save_path  


with open('../temp/CIFAR10_allcnn_training_info.pkl','wb') as output:
    pickle.dump(STEPS,output,pickle.HIGHEST_PROTOCOL)
    pickle.dump(ACC_TRAIN,output,pickle.HIGHEST_PROTOCOL)
    pickle.dump(COST_TRAIN,output,pickle.HIGHEST_PROTOCOL)


FINAL_ACC=0.
for i in xrange(0,10):
    FINAL_ACC+=0.1*sess.run(accuracy, feed_dict={x: CIFAR10.test.images[i*1000:(i+1)*1000], y_: CIFAR10.test.labels[i*1000:(i+1)*1000], keep_prob_input: 1., keep_prob_layers: 1.}) 
   
    
print "Final accuracy on test set:", FINAL_ACC    
    
    
    