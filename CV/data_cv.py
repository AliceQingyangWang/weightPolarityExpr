import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from functools import partial
from random import seed
from random import randint

class data_cv:
    def __init__(self, ds_name, sample_size):
        seed(0) # this is for resetting the seed for every dataset loaded!
        self.ds_name = ds_name 
        self.ds_builder = tfds.builder(ds_name)
        self.info = self.ds_builder.info
        self.ds_builder.download_and_prepare()
        
        self.img_size_orig = self.info.features['image'].shape
        self.train_num = self.info.splits['train'].num_examples
        self.test_num = self.info.splits['test'].num_examples
        self.num_classes = self.info.features['label'].num_classes

        self.sample_size = sample_size
        print(self.sample_size)

        if sample_size == self.train_num: # whole dataset is loaded. makes sense to use it for calculation. Otherwise hard-coded to be consistent
            self.mean, self.std = self.get_norm_param(self.ds_builder.as_dataset(as_supervised=True, split='train')) 
            print('Mean: {mean:1.10f}'.format(mean = self.mean))
            print('Std: {std:1.10f}'.format(std = self.std))
        else:
            if self.ds_name == 'fashion_mnist':
                self.mean = 72.9403522321
                self.std = 90.0211823513
            if self.ds_name == 'cifar10':
                self.mean = [125.3, 123.0, 113.9]
                self.std = [63.0, 62.1, 66.7]
            else:
                print('wrong name!')

    def get_norm_param(self, train_dataset):
        tmp_img = np.array([img[0] for img in list(train_dataset.as_numpy_iterator())])
        return np.mean(tmp_img), np.std(tmp_img)

    def _reverse_order(self, file_instructions):
        return list(reversed(file_instructions))

    def get_train(self):
        read_config =  tfds.ReadConfig(
                            experimental_interleave_sort_fn=self._reverse_order,
                        )
        # if self.sample_size > 999:
        #     print(100*self.sample_size/self.train_num)
        #     return self.ds_builder.as_dataset(as_supervised=True, split = 'train[:{size:d}%]'.format(size = int(100*self.sample_size/self.train_num)), read_config = read_config)
        # else:
        return self.ds_builder.as_dataset(as_supervised=True, split = 'train[:{size:d}]'.format(size = int(self.sample_size)), read_config = read_config)
        # return self.ds_builder.as_dataset(as_supervised=True, split = 'train').take(self.sample_size) # Feel like it will be retaken evey epoch
    
    def get_test(self):
        return self.ds_builder.as_dataset(as_supervised=True, split='test')

    def _process_img(self, image, label, inputSize):
        """Normalizes images: `uint8` -> `float32`."""
        image = (tf.cast(image, tf.float32)-self.mean) / self.std
        # Resize images from 32x32 to 277x277
        image = tf.image.resize(image, (inputSize,inputSize))
        if self.ds_name == 'fashion_mnist':
            image = tf.image.grayscale_to_rgb(image)
        return image, label

    def process_img(self, ds, inputSize):
        return ds.map(partial(self._process_img, inputSize = inputSize), num_parallel_calls=tf.data.AUTOTUNE)

    def prepare_ds(self, ds, inputSize, batchSize, doShuffle):
        total_sample = len(list(ds.as_numpy_iterator()))
        # cache before map here because up-sampling in process_img
        ds = ds.cache(

            )
        if doShuffle: # shuffle has to happen after cache otherwise no effect, it also has to happen before batch, otherwise each batch is exactly the same...
            ds = ds.shuffle(total_sample, seed=randint(0,10), reshuffle_each_iteration=True) #, seed=0 , reshuffle_each_iteration is set to True automatially, should be fine?
        ds = ds.batch(
                batch_size = batchSize
            ).map(
                partial(self._process_img, inputSize = inputSize), num_parallel_calls=tf.data.AUTOTUNE
            ).prefetch(
                tf.data.AUTOTUNE
            )
        return ds       

    def check_threads(self, ds):
        iterator = iter(ds)
        plt.figure(figsize=(20,15))
        for ii in range(12):
            values = next(iterator)
            image = values[0]
            label = values[1]
            ax = plt.subplot(3,4,ii+1)
            ax.imshow(image[0, :,:], cmap = 'gray')
            ax.set_title('{}'.format(label[0]));
    
    def check_range(self, ds):
        tmp_img = np.array([img[0] for img in list(ds.as_numpy_iterator())])
        print('mean is {mean:2.2f}, std is {std:2.2f}, max is {max:2.2f}, min is {min:2.2f}'.format(mean=np.mean(tmp_img), std=np.std(tmp_img), min=np.min(tmp_img), max=np.max(tmp_img))) 