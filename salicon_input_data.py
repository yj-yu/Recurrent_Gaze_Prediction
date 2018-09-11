from __future__ import print_function
from os import listdir
from os.path import isdir, isfile, join
from PIL import Image
import tensorflow as tf
import numpy as np, h5py
import scipy as sp
import scipy.sparse
import sys

import sklearn.model_selection

#sys.path.append('/data/SALICON/PythonAPI')
#from salicon.salicon import SALICON  # is in folder  /gazegap/salion/salicon-evluation/salicon 


def gather_filenames(folder_path):
    return [f for f in listdir(folder_path) if isfile(join(folder_path, f))]


class SaliconDataset(object):
    def __init__(self, images, saliencymaps, fixationmaps=None):
        """
        images : array-like of images (3d ndarray)
        """
        self.images = images
        self.saliencymaps = saliencymaps
        self.fixationmaps = fixationmaps
        self.epochs_completed = 0
        self.index_in_epoch = 0

        assert self.image_count() > 0
        self.batch_perm = list(range(self.image_count()))
        np.random.RandomState(3024202).shuffle(self.batch_perm)

    def __len__(self):
        return self.image_count()

    def image_count(self):
        return len(self.images)

    def __repr__(self):
        return ('<SaliconDataSet with %d images>' % self.image_count())

    def image_shape(self):
        return self.images[0].shape

    def saliencymaps_shape(self):
        return self.saliencymaps[0].shape

    # reference: tensorflow.learn mnist
    def next_batch(self, batch_size):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.image_count():
            # Finished epochs
            self.epochs_completed += 1
            # Shuffle the data
            self.batch_perm = list(range(self.image_count()))
            np.random.shuffle(self.batch_perm)
            #self.images = self.images[perm]
            #self.saliencymaps = self.saliencymaps[perm]
            # Start next epoch
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.image_count()
        end = self.index_in_epoch

        batch_indices = self.batch_perm[start:end]
        return (self.images[batch_indices],
                self.saliencymaps[batch_indices],
                self.fixationmaps[batch_indices])


def read_salicon_data_set(image_folder_path,
                          saliencymap_folder_path,
                          fixationmap_folder_path,
                          image_height, image_width, saliencymap_height,
                          saliencymap_width, dtype=tf.float32):
    """
    Input: path to folders used for wanted data set, eg test, val, train
    Returns: Salicon_dataset 
       
    """
        
    filenames = gather_filenames(image_folder_path)
    filenames.sort()

    images = []
    saliencymaps = []
    fixationmaps = []
    for filename in filenames:
        # input image
        image_filepath = image_folder_path + filename
        image = Image.open(image_filepath).convert('RGB')
        width, height = image.size
        if width != image_width or height != image_height:
            print('WARN: image resizing %s ...' % filename)
            image = image.resize((image_width, image_height), Image.ANTIALIAS)
        image = np.array(image)
        images.append(image)

        # saliency map
        saliencymap_filepath = saliencymap_folder_path + filename
        saliencymap = Image.open(saliencymap_filepath).convert('L')
        width, height = saliencymap.size
        if width != saliencymap_width or height != saliencymap_height:
            print('WARN: saliencymap resizing %s ...' % filename)
            saliencymap = saliencymap.resize((saliencymap_width, saliencymap_height), Image.ANTIALIAS)
        saliencymap = np.array(saliencymap)
        saliencymaps.append(saliencymap)

        # fixation map (in raw scale)
        fixationmap_filepath = fixationmap_folder_path + filename + '.npy'
        fixationmap = np.load(fixationmap_filepath)
        fixationmap = scipy.sparse.csr_matrix(fixationmap, dtype=np.float32) # convert to sparse matrix due to memory limit
        fixationmaps.append(fixationmap)

    images = np.stack(images, axis=0)
    saliencymaps = np.stack(saliencymaps, axis=0)
    fixationmaps = np.array(fixationmaps, dtype=object)

    if dtype == tf.float32:
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)
    saliencymaps = np.stack(saliencymaps, axis=0)
    if dtype == tf.float32:
        saliencymaps = saliencymaps.astype(np.float32)
        saliencymaps = np.multiply(saliencymaps, 1.0 / 255.0)

    return SaliconDataset(images, saliencymaps, fixationmaps)

#
class SaliconData(object):
    
    def __init__(self, image_height, image_width,
                         saliencymap_height,
                         saliencymap_width,
                         dtype=tf.float32,
                         use_example=False,
                         use_val_split=False):

        """
        after init make sure to run data.build()
        for every change to initial variable data.build() needs to be rerun for change to apply
        """
        ## dataset parameters
        self.image_height = image_height
        self.image_width = image_width
        self.saliencymap_height = saliencymap_height
        self.saliencymap_width= saliencymap_width
        self.dtype = dtype
        self.use_example = use_example
        self.use_val_split = use_val_split
        self.train = None
        self.valid = None 
        self.test = None

    def build(self):
        """
        Returns: loads validation,test and training dataset
        """
        print("loading data set ...")

        
        SALICON_PATH = 'salicon/'
        
        VALIDATION_IMAGE = SALICON_PATH + 'images/val98x98/'
        VALIDATION_SALIENCYMAP = SALICON_PATH + 'saliencymaps/val49x49/'
        VALIDATION_FIXATIONMAP = SALICON_PATH + 'fixations/val/'
        
        if self.use_example:
            image_folder_path = SALICON_PATH + 'images/train2014examples/'
            saliencymap_folder_path = SALICON_PATH + 'saliencymaps/train2014examples/'
            fixationmap_folder_path = SALICON_PATH + 'fixations/train2014examples/'
        else:
            image_folder_path = SALICON_PATH + 'images/train98x98/'
            saliencymap_folder_path = SALICON_PATH + 'saliencymaps/train49x49/'
            fixationmap_folder_path = SALICON_PATH + 'fixations/train/'
            
            self.train = read_salicon_data_set(image_folder_path,
                                                    saliencymap_folder_path,
                                                    fixationmap_folder_path,
                                                    self.image_height, self.image_width,
                                                    self.saliencymap_height, self.saliencymap_width,
                                                    self.dtype)

            # SALICON dataset does not provide public TEST set
            # so, for this experiment, we use split the Validation set and use the split as Test set
        if not self.use_example:
            self.test = read_salicon_data_set(
                VALIDATION_IMAGE, VALIDATION_SALIENCYMAP, VALIDATION_FIXATIONMAP,
                self.image_height, self.image_width, self.saliencymap_height, self.saliencymap_width,
                self.dtype
            )

            # modified
        if self.use_val_split:
            all_dataset = self.train
            (images_train, images_test, \
             saliencymaps_train, saliencymaps_test, \
             fixationmaps_train, fixationmaps_test, \
            ) = sklearn.model_selection.train_test_split(all_dataset.images,
                                                          all_dataset.saliencymaps,
                                                          all_dataset.fixationmaps,
                                                          test_size=0.2)
            self.train = SaliconDataset(images_train, saliencymaps_train, fixationmaps_train)
            self.valid = SaliconDataset(images_test, saliencymaps_test, fixationmaps_test)
        else:
            self.valid = self.test
            
        print("Done.")
        


if __name__ == '__main__':
    global data_sets #why is this global?
    salicon = SaliconData(98, 98, 49, 49)
    data_sets = salicon.build()

    
