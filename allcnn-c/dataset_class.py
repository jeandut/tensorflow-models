# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 10:43:13 2016

@author: jeandut
"""
import numpy as np
import tensorflow as tf
from scipy import ndimage


#This is me reimplementing dataset_class which can be found in tensorflow MNIST tutorial
class DataSet(object):

  def __init__(self, images, labels, preprocess="scale", da=False,
               dtype=tf.float32):

    dtype = tf.as_dtype(dtype).base_dtype
    if dtype not in (tf.uint8, tf.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)

      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape,
                                                 labels.shape))
      
    self._num_examples = images.shape[0]


    if dtype == tf.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)
        
        if preprocess=="IMAGENET":
            
            VGG_MEAN=np.array([123.68, 116.779, 103.99])
            
            images[:,:,:,0]-= VGG_MEAN[0]
            images[:,:,:,1]-= VGG_MEAN[1]
            images[:,:,:,2]-= VGG_MEAN[2]
            
        elif preprocess=="scale":            
            images = np.multiply(images, 1.0 / 255.0)
        
    self._images = images
    self._labels = labels
    self._DA=da
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels
    
  @property
  def DA(self):
    return self._DA

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed
    



  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
   
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    
    
    

    
    return self.images[start:end], self._labels[start:end]


def read_data_sets(train_images, train_labels, test_images, test_labels, preprocess="scale", DA=False, VALIDATION_SIZE=0, dtype=tf.float32):
  class DataSets(object):
    pass
  data_sets = DataSets()



  validation_images = train_images[:VALIDATION_SIZE]
  validation_labels = train_labels[:VALIDATION_SIZE]
  train_images = train_images[VALIDATION_SIZE:]
  train_labels = train_labels[VALIDATION_SIZE:]



  
#Creation de l'objet dataset

  data_sets.train = DataSet(train_images, train_labels, preprocess=preprocess, da=DA, dtype=dtype)
  data_sets.validation = DataSet(validation_images, validation_labels, preprocess=preprocess, da=DA, dtype=dtype)
  data_sets.test = DataSet(test_images, test_labels, preprocess=preprocess, da=DA, dtype=dtype)

  return data_sets
