# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 11:20:51 2017

@author: jeandut
"""
#On charge les librairies dont on a besoin i.e; tensorflow et numpy TOUJOURS +librairies accessoires on leur donne des petits noms pour ne pas réecrire matplotlib.pyplot.method à chaque fois
import tensorflow as tf
import numpy as np
from gen_batches_siamese import train_batches_gen
import matplotlib.pyplot as plt
import re


#On import les data depuis les sauvegardes numpy
train_images=np.load("train_images.npy")
test_images=np.load("test_images.npy")
train_labels=np.load("train_labels.npy")
test_labels=np.load("test_labels.npy")
#Construire un réseau siamois 
#Première étape je créé proprement les placeholders dont je vais avoir besoin (attention c'est un réseau SIAMOIS) j'utilise tf.placeholder(tf.float32,[à vous de trouver],name="votre_choix_de_nom") indice il y en a 4

left_branch_input=tf.placeholder(tf.float32,[None,28,28,1],name="left_branch_input") #Notez la dimension implicite
right_branch_input=tf.placeholder(tf.float32,[None,28,28,1],name="right_branch_input")
left_branch_label=tf.placeholder(tf.float32,[None,1],name="left_branch_label")
right_branch_label=tf.placeholder(tf.float32,[None,1],name="right_branch_label")
#corrigé

#Je procède linéairement en commencant par la première branche et en suivant les spécifications du réseau je crée variables et poids que j'initialise à 0. j'utilise tf.nn.conv2d() en regardant la doc

def weights(kernel_shape):
    if len(kernel_shape)==2:
        fan_in=kernel_shape[0]
        fan_out=kernel_shape[1]
    else:
        fan_in=np.prod(kernel_shape[:3])
        fan_out=kernel_shape[3]
    a=np.sqrt(6. /(fan_in+fan_out))   
    weights=tf.get_variable("weights",kernel_shape,initializer=tf.random_uniform_initializer(minval=-a,maxval=a))
    return weights
    
def biases(bias_shape):
    bias=tf.get_variable("biases",bias_shape,initializer=tf.constant_initializer(0.))
    return bias
    
def create_branch(branch_input, reuse_arg=False):
    with tf.variable_scope("conv1", reuse=reuse_arg):
        w=weights((5,5,1,20))
        b=biases((20))
        branch=tf.nn.conv2d(branch_input,w,[1,1,1,1],"SAME")
        branch=tf.add(branch,b)
        branch=tf.nn.relu(branch)
    
    branch=tf.nn.max_pool(branch,[1,2,2,1],[1,2,2,1],"VALID",name="pool1")
    
    with tf.variable_scope("conv2", reuse=reuse_arg):
        w=weights((5,5,20,50))
        b=biases((50))
        branch=tf.nn.conv2d(branch,w,[1,1,1,1],"SAME")
        branch=tf.add(branch,b)
        branch=tf.nn.relu(branch)
    
    branch=tf.nn.max_pool(branch,[1,2,2,1],[1,2,2,1],"VALID",name="pool2")
    
    flattened_shape=np.prod(branch.get_shape().as_list()[1:])
    
    branch=tf.reshape(branch,[-1,flattened_shape],"flatten")
    
    with tf.variable_scope("fc1", reuse=reuse_arg):
        w=weights((flattened_shape,500))
        b=biases((500))
        branch=tf.matmul(branch,w)
        branch=tf.add(branch,b)
        branch=tf.nn.relu(branch)
    
    with tf.variable_scope("fc2", reuse=reuse_arg):
        w=weights((500,10))
        b=biases((10))
        branch=tf.matmul(branch,w)
        branch=tf.add(branch,b)
        branch=tf.nn.relu(branch)
    
    with tf.variable_scope("feature", reuse=reuse_arg):
        w=weights((10,2))
        b=biases((2))
        branch=tf.matmul(branch,w)
        branch=tf.add(branch,b)
    return branch
    
#Faire la seconde branche en utilisant le même code!
left_branch=create_branch(left_branch_input, False)
right_branch=create_branch(right_branch_input, True)



#Vérifiez que les deux branches sont de dimensions None,2 si vous ne vous êtes pas trompés (un coup de sortie1.get_shape().as_list() ==[None,2] et sortie1.get_shape()==sortie2.get_shape() pour s'en assurer
#Avant de recombiner les branches dans le côut il nous manque le label=1 si class_1==class_2 0 sinon pour cela utiliser les fonctions tf.equal() qui donne un booléen et tf.cast()

with tf.name_scope("label_pair"):
    is_same_class=tf.equal(left_branch_label, right_branch_label)
    is_same_class=tf.cast(is_same_class,tf.float32)
    ss=tf.scalar_summary("same_pair_proportion_in_batch",tf.reduce_mean(is_same_class))
    
#Il nous manque aussi la distance L2 entre les deux embeddings (utilisez d'autres distances si vous avez l'impression que ce tutorial est trop facile) il faut pour cela utiliser un reduce_sum un tf.sqrt() et un tf.square()

with tf.name_scope("squared_distance_between_features"):
    D_2=tf.reshape(tf.reduce_sum(tf.square(left_branch-right_branch), reduction_indices=1),(-1,1))
    
with tf.name_scope("loss_same_pair"):
    cost_same_pair=is_same_class*0.5*D_2
    ss=tf.scalar_summary("cost_same_pair",tf.reduce_mean(cost_same_pair))
    
    
with tf.name_scope("loss_different_pair"):
    cost_diff_pair=(1.-is_same_class)*0.5*tf.square(tf.nn.relu(1.-tf.sqrt(D_2)))
    ss=tf.scalar_summary("cost_diff_pair",tf.reduce_mean(cost_diff_pair))
    
    
with tf.name_scope("contrastive_loss"):
    cost=tf.reduce_mean(cost_same_pair+cost_diff_pair)
    ss=tf.scalar_summary("contrastive_loss",cost)

with tf.name_scope("learning_rate_decay"):
    #On reproduit le decay lr_inv_policy de caffe
    global_step=tf.Variable(0,trainable=False)
    power=tf.constant(0.75)
    gamma=tf.constant(0.0001)
    starter_lr=tf.constant(0.01)
    lr=starter_lr*tf.pow(1.+gamma*tf.cast(global_step,tf.float32),-power)
    
    
with tf.name_scope("optimization"):
    opt=tf.train.MomentumOptimizer(lr,0.9)
    grads_and_vars=opt.compute_gradients(cost,tf.trainable_variables())
    #Passage un peu obscur pour reproduire l'ffet de multiplication du gradient par deux sur les biais de l'implémentation Caffe
    grads_and_vars_biases=[(2*gv[0],gv[1]) for gv in grads_and_vars if "biases" in gv[1].name]
    grads_and_vars_weights=[(gv[0],gv[1]) for gv in grads_and_vars if "weights" in gv[1].name ]
    grads_and_vars=grads_and_vars_weights+grads_and_vars_biases
#    for gv in grads_and_vars:
#        ss=tf.scalar_summary("GRAD_MEAN_"+re.sub(":0","",gv[1].name),tf.reduce_mean(gv[0]))
#        ss=tf.scalar_summary("GRAD_MIN_"+re.sub(":0","",gv[1].name),tf.reduce_min(gv[0]))
#        ss=tf.scalar_summary("GRAD_MAX"+re.sub(":0","",gv[1].name),tf.reduce_max(gv[0]))

    train_step=opt.apply_gradients(grads_and_vars,global_step)
    
merged=tf.merge_all_summaries()

summaries_only_for_train=[]
ss=tf.scalar_summary("learning_rate",lr)
summaries_only_for_train.append(ss)
for var in tf.trainable_variables():
    basename=re.sub(":0","",var.name)
    hs=tf.histogram_summary("HIST_"+basename,var)
    summaries_only_for_train.append(hs)
    mean=tf.reduce_mean(var)
    ss=tf.scalar_summary("MEAN_"+basename,mean)
    summaries_only_for_train.append(ss)
    ss=tf.scalar_summary("STDDEV_"+basename,tf.reduce_mean(tf.square(var-mean)))
    summaries_only_for_train.append(ss)
    ss=tf.scalar_summary("MIN_"+basename,tf.reduce_min(var))
    summaries_only_for_train.append(ss)
    ss=tf.scalar_summary("MAX_"+basename,tf.reduce_max(var))
    summaries_only_for_train.append(ss)
    
    if ("conv1/weights" in var.name):
        #C'est moche mais avec Tensorboard on ne peut pas vraiment faire mieux (rajouter des colonnes et lignes de pixels noir pour délimiter les différents poids manuellement est laissé à la discretion du lecteur)
        painting=tf.split(3,20,var)         
        painting_row0=tf.concat(0,painting[0:5])
        painting_row1=tf.concat(0,painting[5:10])
        painting_row2=tf.concat(0,painting[10:15])
        painting_row3=tf.concat(0,painting[15:20])
    
    
    
        painting=tf.concat(1,[painting_row0,painting_row1,painting_row2,painting_row3])
        painting=tf.reshape(painting,[1,25,20,1])
        isu=tf.image_summary("grid_of_"+basename,painting)
        summaries_only_for_train.append(isu)
        
        
    

    


saver=tf.train.Saver()

sess=tf.Session()
sess.run(tf.initialize_all_variables())
train_writer=tf.train.SummaryWriter("./train/",sess.graph)
test_writer=tf.train.SummaryWriter("./test/",sess.graph)
#On import les data depuis les sauvegardes numpy
train_images=np.load("train_images.npy")
test_images=np.load("test_images.npy")
train_labels=np.load("train_labels.npy")
test_labels=np.load("test_labels.npy")

#On preprocess les images du train comme du test en les normalisant entre 0. et 1. en valeur float32 (single precision) attention à convertir le tableau avant de diviser en float
#Puis ATTENTION VERIFIER QUE LES DIMENSIONS D'UN BATCH SONT BIEN CELLES QU'ON VEUT !!! (indice ce n'est pas le cas...) Attention aux dimensions 1 !
train_labels, train_images, test_labels,test_images=train_labels[:,None], train_images[:,:,:,None], test_labels[:,None],test_images[:,:,:,None]

train_images=train_images.astype("float32")/255.
test_images=test_images.astype("float32")/255.
BATCH_SIZE=32
#nb_epochs=100
#num_batches_per_epoch=train_images.shape[0]//(2*BATCH_SIZE)
#nb_iter=nb_epochs*num_batches_per_epoch
nb_iter=50000

gen_train=train_batches_gen(train_images,train_labels)
#On prend aussi les premiers 1000 exemples du test pour vérifier qu'on overfitte pas
gen_test=train_batches_gen(test_images,test_labels)
test=gen_test.next_batch(1000)

for i in xrange(nb_iter):
    batch_images_lb, batch_labels_lb, batch_images_rb, batch_labels_rb=gen_train.next_batch(BATCH_SIZE)
    feed_dict={left_branch_input: batch_images_lb, left_branch_label: batch_labels_lb, right_branch_input:batch_images_rb, right_branch_label:batch_labels_rb}
    if (i%100):
        m,summaries=sess.run([merged,summaries_only_for_train],feed_dict=feed_dict)
        train_writer.add_summary(m,i)
        for s in summaries:
            train_writer.add_summary(s,i)

    if (i%500==0):
        feed_dict_test={left_branch_input: test[0], left_branch_label: test[1], right_branch_input:test[2], right_branch_label:test[3]}
        summaries=sess.run(merged,feed_dict=feed_dict_test)
        test_writer.add_summary(summaries,i)
        saver.save(sess,"./train/model")
        
    sess.run(train_step,feed_dict=feed_dict)


#On calcule les embeddings du test set (par batch sinon ca ne va probablement pas rentrer en mémoire enfin vous pouvez essayer...) 
embeddings=np.empty((test_images.shape[0],2))
for i in xrange(10):
    embeddings[i*100:(i+1)*100,:]=sess.run(left_branch,feed_dict={left_branch_input: test_images[i*100:(i+1)*100,:,:,:]})
    
#On va plotter sur le même graphique les positions des différents embeddings et les colorer en fonction de la classe associée si l'entrainement est réussi les points de la même classe devraient occuper la même région de l'espace
    
f = plt.figure(figsize=(16,9))
c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff','#ff00ff', '#990000', '#999900', '#009900', '#009999']
for i in xrange(10):
    embeddings_current_class=embeddings[test_labels[:,0]==i]
    plt.plot(embeddings_current_class[:,0],embeddings_current_class[:,1],".",label="class %s"%i, color=c[i])
plt.legend()
plt.title("MNIST embeddings visualization")
plt.show()

    




