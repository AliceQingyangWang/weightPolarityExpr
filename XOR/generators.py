from tabnanny import verbose
import numpy as np
import matplotlib.pyplot as plt
import math as math
import tensorflow as tf
import os
import pickle

def generateXOR(num_sample, num_dim):
    num_sample_per_gp = int(np.floor(num_sample/4));
    actual_num_sample = num_sample_per_gp*4;

    gp1 = np.random.rand(num_sample_per_gp, 2);
    gp3 = np.random.rand(num_sample_per_gp, 2)+1.05;
    gp2 = np.random.rand(num_sample_per_gp, 2);
    gp2[:,1]+=1.05;
    gp4 = np.random.rand(num_sample_per_gp, 2);
    gp4[:,0]+=1.05;

    x = np.reshape(np.array([gp1[:,0], gp3[:,0], gp2[:,0], gp4[:,0]]), (1,-1))-1;
    y = np.reshape(np.array([gp1[:,1], gp3[:,1], gp2[:,1], gp4[:,1]]), (1,-1))-1;
    highD = np.random.normal(loc=1.0, scale=1.0, size=(num_dim-2, x.shape[1]))
    data = np.concatenate((x, y, highD), axis=0)

    labels = np.concatenate((np.ones(num_sample_per_gp*2),np.zeros(num_sample_per_gp*2)));
    dataset = {}
    dataset['data'] = data
    dataset['labels'] = labels

    return dataset, actual_num_sample

def supplementXOR(dataset_in, num_add_sample):
    if num_add_sample > 0:
        num_dim = dataset_in['data'].shape[0]
        add_dataset = generateXOR(num_add_sample, num_dim)[0]
        dataset_out = {}
        dataset_out['data'] = np.concatenate((dataset_in['data'], add_dataset['data']), axis=1)
        dataset_out['labels'] = np.concatenate((dataset_in['labels'], add_dataset['labels']))
        return dataset_out, dataset_out['labels'].shape[0]
    else:
        return dataset_in, dataset_in['labels'].shape[0]

def tensorize_datasets(dataset_in, batch_size, do_shuffle):
    total_sample = dataset_in['data'].shape[1]
    ds = tf.data.Dataset.from_tensor_slices((dataset_in['data'].transpose(), dataset_in['labels']))    
    ds_tensor = ds.cache()
    if do_shuffle:
        ds_tensor = ds_tensor.shuffle(total_sample, seed=0)
    ds_tensor = ds_tensor.batch(batch_size)
    ds_tensor = ds_tensor.prefetch(tf.data.AUTOTUNE)
    return ds_tensor
    # print(len(list(ds.as_numpy_iterator())))

from helperfun import build_string_from_dict, get_dict_by_key

# generate all samples and save them to pickle files
def generate_single_xper_2d(sample_size, sample_size_val, config):
    dataset_train, num_sample_base = generateXOR(sample_size, config['num_dim'])

    if sample_size_val == 4: #TODO
        dataset_val = {}
        dataset_val['data'] = np.transpose(np.array([[.5,.5], [-.5,-.5], [.5,-.5], [-.5,.5]]))
        dataset_val['labels'] = np.array([1,1,0,0])
    else:
        dataset_val, num_sampleval = generateXOR(sample_size_val, config['num_dim'])

    config['sample_size'] = num_sample_base
    save_sample(dataset_train, dataset_val, config)
        
    del config['sample_size']

# generate all samples and save them to pickle files
def generate_series_xper_samples(sample_size_list, sample_size_val, config):
    sample_size_diff = np.insert(np.diff(sample_size_list), 0, 0)

    dataset_train, num_sample_base = generateXOR(sample_size_list[0], config['num_dim'])
    dataset_val, num_sample_val = generateXOR(sample_size_val, config['num_dim'])

    config['sample_size'] = num_sample_base
    sample_size_list[0] = num_sample_base

    for idx, diff_sample_size in enumerate(sample_size_diff):
        dataset_train, total_sample = supplementXOR(dataset_train, diff_sample_size)
        config['sample_size'] = total_sample
        sample_size_list[idx] = total_sample

        save_sample(dataset_train, dataset_val, config)
        
    del config['sample_size']

def save_sample(dataset_train, dataset_val, config):
    f_name = os.path.join(os.getcwd(), 'data', build_string_from_dict(get_dict_by_key(config, {'num_dim', 'sample_size'})) + '.pkl')
    if not os.path.isdir(os.path.join(os.getcwd(), 'data')):
        os.makedirs(os.path.join(os.getcwd(), 'data'))
    with open(f_name, 'wb+') as f:
        pickle.dump({'dataset_train':dataset_train, 'dataset_val':dataset_val}, f)

def generate_datasets(num_dim, batch_size, validation_sample = 1000, verbose = False, **kwargs):
    if 'dataset_train' in kwargs:
        dataset = kwargs['dataset_train']
        total_sample = kwargs['dataset_train']['data'].shape[1]
    else:
        total_sample = kwargs['total_sample']
        dataset, total_sample = generateXOR(total_sample, num_dim)
    ds_train = tensorize_datasets(dataset, batch_size, True)

    # always generate 1000 validation points
    dataset_val, validation_sample = generateXOR(validation_sample, num_dim)
    ds_val = tensorize_datasets(dataset_val, batch_size, False)

    return ds_train, ds_val

def generate_datasets_threaded(num_dim, batch_size, num_threads, validation_sample = 1000, verbose = False, **kwargs):
    # this will create two exactly the same ds_train so to be used for exact same training samples for two networks
    if 'dataset_train' in kwargs:
        dataset = kwargs['dataset_train']
        total_sample = kwargs['dataset_train']['data'].shape[1]
    else:
        total_sample = kwargs['total_sample']
        dataset, total_sample = generateXOR(total_sample, num_dim)

    ds_train_threaded = []
    for tt in range(num_threads): 
        ds_train = tensorize_datasets(dataset, batch_size, True)
        ds_train_threaded.append(ds_train)   
    
    # always generate 1000 validation points
    dataset_val, validation_sample = generateXOR(validation_sample, num_dim)
    ds_val = tensorize_datasets(dataset_val, batch_size, False)

    return ds_train_threaded, ds_val

def plotXOR(features, labels):
    x = features[0,:]
    y = features[1,:]
    if features.shape[0]-1 > 2:
        z = features[np.random.randint(2, features.shape[0]-1), :]
    elif features.shape[0] < 3:
        z = []
    else:
        z = features[2, :]
    colors = labels

    fig = plt.figure(figsize = (10, 7))
    if len(z) < 1:
        ax = plt.gca()
        sc = ax.scatter(x, y, c=colors, alpha = 1, s = 5, cmap = 'rainbow')
        ax.plot([0,0], [-1,1.1], 'k--');
        ax.plot([-1,1.1], [0,0], 'k--');
    else:    
        ax = plt.axes(projection ="3d")
        sc = ax.scatter(x, y, z, c=colors, alpha = 1, s = 5, cmap = 'rainbow')
        ax.plot([0,0], [-1,1.1], [1,1], 'k--');
        ax.plot([-1,1.1], [0,0], [1,1], 'k--');

    ax.set_xticks(np.arange(-1,1,0.5));
    ax.set_yticks(np.arange(-1,1,0.5));

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$');
    plt.colorbar(sc)