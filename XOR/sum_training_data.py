from cmath import nan
from glob import glob, os
from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorflow.python.framework import tensor_util
import numpy as np
import pandas as pd
import psutil
from multiprocessing import Pool
from functools import partial

def get_row(array, keys, numRun, baseFName=''):
    this_row = dict(zip(keys, array))
    num_epoch = this_row['num_epoch']
    for val_train in ['train', 'validation']:
        log_dir = os.path.join(os.getcwd(), baseFName, 'dim%d' % this_row['num_dim'], this_row['resetType'],
                            'logs', 's%de%dr%d' % (this_row['train_sample'], this_row['num_epoch'], this_row['hiddenSize']), 
                            "*_%s" % this_row['typeStr'], val_train, 'events*')
        pathList = glob(log_dir)
        epoch_acc = np.empty(shape=(numRun, num_epoch), dtype = np.float)
        epoch_acc[:] = nan
        epoch_loss = np.empty(shape=(numRun, num_epoch), dtype = np.float)
        epoch_loss[:] = nan
        pre_epoch_acc = np.empty(shape=(numRun, 2), dtype = np.float)
        pre_epoch_acc[:] = nan

        for thisPath in pathList:
            thisRun = int(thisPath.split('/')[-3].split('_')[0][1:])
            if thisRun < numRun:
                # print(thisRun)
                for e in summary_iterator(thisPath):
                    for v in e.summary.value:
                        if v.tag == 'epoch_accuracy':
                            t = tensor_util.MakeNdarray(v.tensor)
                            if e.step < 0:
                                pre_epoch_acc[thisRun, e.step+2] = t
                            else:
                                epoch_acc[thisRun, e.step] = t
                        if v.tag == 'epoch_loss':
                            t = tensor_util.MakeNdarray(v.tensor)
                            epoch_loss[thisRun, e.step] = t
        this_row[val_train+'_acc'] = [epoch_acc]
        this_row[val_train+'_loss'] = [epoch_loss]
        this_row['pre_'+val_train+'_acc'] = [pre_epoch_acc]
    return pd.DataFrame(this_row, index=[0])


def getMetrics_parallel(num_dim_list, resetType_list, hSize_list, num_epoch_list, sample_size_list, doWeightFreeze_list, numRun, baseFName = '', **kwargs):
    easy_index_cal_list = np.array([len(num_dim_list), len(resetType_list), len(hSize_list), len(sample_size_list), len(doWeightFreeze_list)])
    total_num_rows = np.prod(easy_index_cal_list)
    metrics = pd.DataFrame(index=range(total_num_rows), columns=["num_dim", "resetType", "hiddenSize", "train_sample", "num_epoch", "typeStr", "train_acc", 'train_loss', 'validation_acc', 'validation_loss', 'pre_train_acc', 'pre_validation_acc'])
    row_accumulator = 0

    for num_dim in num_dim_list:
        metrics['num_dim'][row_accumulator+np.arange(np.prod(easy_index_cal_list[1:]))] = num_dim

        for resetType in resetType_list:
            metrics['resetType'][row_accumulator+np.arange(np.prod(easy_index_cal_list[2:]))] = resetType

            for hiddenSize in hSize_list: 
                metrics['hiddenSize'][row_accumulator+np.arange(np.prod(easy_index_cal_list[3:]))] = hiddenSize

                for (train_sample, num_epoch) in zip(sample_size_list, num_epoch_list):
                    metrics['train_sample'][row_accumulator+np.arange(np.prod(easy_index_cal_list[4:]))] = train_sample
                    metrics['num_epoch'][row_accumulator+np.arange(np.prod(easy_index_cal_list[4:]))] = num_epoch

                    for doWeightFreeze in doWeightFreeze_list:
                        if doWeightFreeze:
                            typeStr = 'freeze'
                        else:
                            typeStr = 'liquid'
                        metrics['typeStr'][row_accumulator] = typeStr
                        row_accumulator+=1

    num_cpus = psutil.cpu_count(logical=False)
    part_get_row = partial(get_row, keys=metrics.columns.tolist(), numRun=numRun, baseFName=baseFName)
    # Create the pool
    with Pool(processes=num_cpus) as process_pool:
        # Start processes in the pool
        dfs = process_pool.map(part_get_row, metrics.to_numpy().tolist())
        # Concat dataframes to one dataframe
        data = pd.concat(dfs, ignore_index=True)
        return data


def getMetrics(num_dim_list, resetType_list, hSize_list, num_epoch_list, sample_size_list, doWeightFreeze_list, numRun, **kwargs):
    easy_index_cal_list = np.array([len(num_dim_list), len(resetType_list), len(hSize_list), len(sample_size_list), len(doWeightFreeze_list)])
    total_num_rows = np.prod(easy_index_cal_list)
    metrics = pd.DataFrame(index=range(total_num_rows), columns=["num_dim", "resetType", "hiddenSize", "train_sample", "typeStr", "train_acc", 'train_loss', 'validation_acc', 'validation_loss', 'pre_train_acc', 'pre_validation_acc'])
    row_accumulator = 0

    for num_dim in num_dim_list:
        metrics['num_dim'][row_accumulator+np.arange(np.prod(easy_index_cal_list[1:]))] = num_dim
        baseFName = 'dim%d' % num_dim

        for resetType in resetType_list:
            metrics['resetType'][row_accumulator+np.arange(np.prod(easy_index_cal_list[2:]))] = resetType

            for hiddenSize in hSize_list: 
                metrics['hiddenSize'][row_accumulator+np.arange(np.prod(easy_index_cal_list[3:]))] = hiddenSize

                for (train_sample, num_epoch) in zip(sample_size_list, num_epoch_list):
                    metrics['train_sample'][row_accumulator+np.arange(np.prod(easy_index_cal_list[4:]))] = train_sample

                    for doWeightFreeze in doWeightFreeze_list:
                        if doWeightFreeze:
                            typeStr = 'freeze'
                        else:
                            typeStr = 'liquid'
                        metrics['typeStr'][row_accumulator] = typeStr

                        for val_train in ['train', 'validation']:
                            log_dir = os.path.join(os.getcwd(), baseFName, resetType, 'logs', 's%de%dr%d' % (train_sample, num_epoch, hiddenSize), "*_%s" % typeStr, val_train, 'events*')
                            pathList = glob(log_dir)
                            epoch_acc = np.empty(shape=(numRun, num_epoch), dtype = np.float)
                            epoch_acc[:] = nan
                            epoch_loss = np.empty(shape=(numRun, num_epoch), dtype = np.float)
                            epoch_loss[:] = nan
                            pre_epoch_acc = np.empty(shape=(numRun, 2), dtype = np.float)
                            pre_epoch_acc[:] = nan

                            for thisPath in pathList:
                                thisRun = int(thisPath.split('/')[-3].split('_')[0][1:])
                                # print(thisRun)
                                for e in summary_iterator(thisPath):
                                    for v in e.summary.value:
                                        if v.tag == 'epoch_accuracy':
                                            t = tensor_util.MakeNdarray(v.tensor)
                                            if e.step < 0:
                                                pre_epoch_acc[thisRun, e.step+2] = t
                                            else:
                                                epoch_acc[thisRun, e.step] = t
                                        if v.tag == 'epoch_loss':
                                            t = tensor_util.MakeNdarray(v.tensor)
                                            epoch_loss[thisRun, e.step] = t
                            metrics[val_train+'_acc'][row_accumulator] = [epoch_acc]
                            metrics[val_train+'_loss'][row_accumulator] = [epoch_loss]
                            metrics['pre_'+val_train+'_acc'][row_accumulator] = [pre_epoch_acc]

                        row_accumulator+=1
            print('network width %d is done!' % hiddenSize)
    return metrics