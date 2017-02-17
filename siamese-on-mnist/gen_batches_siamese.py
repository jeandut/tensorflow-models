import numpy as np

class train_batches_gen(object):
    def __init__(self, train_images, train_labels):
        assert train_images.shape[1:]==(28,28,1), "The images do not meet the shape requirements of tensorflow"
        assert train_labels.shape[0]==train_images.shape[0], "The labels and images do not have the same first dimension. It is suspicious."
        assert train_images.max()==1. and train_images.min()==0., "You have forgotten to normalize the data"
        self._images=train_images
        self._labels=train_labels
        self._index_in_epoch=0
        self._epochs_completed=0
        self._num_examples=train_images.shape[0]
        
    def __iter__(self):
        return self
    
    def next_batch(self, batch_size):
        
        assert 2*batch_size <= self._num_examples
        start = self._index_in_epoch
        self._index_in_epoch += 2*batch_size
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
            self._index_in_epoch = 2*batch_size
           
       
        end = self._index_in_epoch
       
        batch_left_images=self._images[start:start+batch_size]
        batch_left_labels=self._labels[start:start+batch_size]
        batch_right_images=self._images[start+batch_size:end]
        batch_right_labels=self._labels[start+batch_size:end]
        
        return batch_left_images, batch_left_labels, batch_right_images, batch_right_labels
