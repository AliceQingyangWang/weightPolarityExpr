from cmath import nan
from glob import glob, os
from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorflow.python.framework import tensor_util
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd
import psutil
from multiprocessing import Pool
from functools import partial

def get_row(array, keys, numRun, baseFName=''):
    this_row = dict(zip(keys, array))
    num_epoch = this_row['num_epoch']
    for val_train in ['train', 'validation']:
        log_dir = os.path.join(os.getcwd(), baseFName, this_row['resetType'],
                            'logs', 's%de%d_%s' % (this_row['train_sample'], 50, this_row['networkType']), 
                            "*_%s" % this_row['typeStr'], val_train, 'events*') #this_row['num_epoch'] #TODO
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
                        if e.step < num_epoch:
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


def getMetrics_parallel(resetType_list, networkType_list, num_epoch_list, sample_size_list, doWeightFreeze_list, numRun, baseFName = '', **kwargs):
    easy_index_cal_list = np.array([len(resetType_list), len(networkType_list), len(sample_size_list), len(doWeightFreeze_list)])
    total_num_rows = np.prod(easy_index_cal_list)
    metrics = pd.DataFrame(index=range(total_num_rows), columns=["resetType", "networkType", "train_sample", "num_epoch", "typeStr", "train_acc", 'train_loss', 'validation_acc', 'validation_loss', 'pre_train_acc', 'pre_validation_acc'])
    row_accumulator = 0

    for resetType in resetType_list:
        metrics['resetType'][row_accumulator+np.arange(np.prod(easy_index_cal_list[1:]))] = resetType

        for networkType in networkType_list: 
            metrics['networkType'][row_accumulator+np.arange(np.prod(easy_index_cal_list[2:]))] = networkType

            for (train_sample, num_epoch) in zip(sample_size_list, num_epoch_list):
                metrics['train_sample'][row_accumulator+np.arange(np.prod(easy_index_cal_list[3:]))] = train_sample
                metrics['num_epoch'][row_accumulator+np.arange(np.prod(easy_index_cal_list[3:]))] = num_epoch

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

def sum_plotting(whole_list, num_epoch_list, sample_size_list, resetType_list, xstr, ystr, fig, showYaxis):
    num_t_plot = len(sample_size_list)
    # fig = plt.figure(figsize=(num_t_plot*3,num_t_plot*2))
    # rect = fig.patch
    # rect.set_facecolor([0.6,0.6,0.6])

    gs_w = 0.33
    heading_increment = (1-gs_w)/(len(resetType_list)-1)
    gs_divide_lines = np.concatenate([[0], np.cumsum(np.ones(len(resetType_list)-1)*heading_increment), [1]])
    # gs = gridspec.GridSpec(1,len(resetType_list))
    if isinstance(whole_list, dict):
        ymin = min([min([np.amin(np.apply_along_axis(np.nanpercentile, 0, x_toplot, 25)) for x_toplot in x_toplot_list]) for x_toplot_list in whole_list['freeze']])
        ymin = min([ymin, min([min([np.amin(np.apply_along_axis(np.nanpercentile, 0, x_toplot, 25)) for x_toplot in x_toplot_list]) for x_toplot_list in whole_list['liquid']])])
        ymax = max([max([np.amax(np.apply_along_axis(np.nanpercentile, 0, x_toplot, 75)) for x_toplot in x_toplot_list]) for x_toplot_list in whole_list['freeze']])
        ymax = max([ymax, max([max([np.amax(np.apply_along_axis(np.nanpercentile, 0, x_toplot, 75)) for x_toplot in x_toplot_list]) for x_toplot_list in whole_list['freeze']])])
    else:
        ymin = min([min([np.amin(np.apply_along_axis(np.nanpercentile, 0, x_toplot, 25)) for x_toplot in x_toplot_list]) for x_toplot_list in whole_list])
        ymax = max([max([np.amax(np.apply_along_axis(np.nanpercentile, 0, x_toplot, 75)) for x_toplot in x_toplot_list]) for x_toplot_list in whole_list])
    yrange = [ymin, ymax]

    for idx, folderName in enumerate(resetType_list):
        titlestr = folderName
        gs = gridspec.GridSpec(num_t_plot,num_t_plot*2)

        gs.update(left = gs_divide_lines[idx], right = gs_divide_lines[idx]+gs_w, bottom=0, top=1)
        if isinstance(whole_list, dict):
            x_toplot_list = whole_list['freeze'][idx]
            ax_objs = plot_ridge_series(x_toplot_list, num_epoch_list, sample_size_list, yrange, titlestr, xstr, ystr, fig, gs, showYaxis)
            
            cmap = plt.get_cmap('viridis', len(sample_size_list))
            colors = cmap.colors
            for idxa, (thisAx, x_toplot) in enumerate(zip(ax_objs, whole_list['liquid'][idx])):
                num_epoch = num_epoch_list[idxa]
                mu = np.apply_along_axis(np.nanmedian, 0, x_toplot)
                lower = np.apply_along_axis(np.nanpercentile, 0, x_toplot, 25)
                upper = np.apply_along_axis(np.nanpercentile, 0, x_toplot, 75)
                # plotting the distribution
                thisAx.plot(range(1,num_epoch+1), mu, '--', color=colors[idxa])
                thisAx.fill_between(range(1,num_epoch+1), lower, upper, color=colors[idxa], alpha=0.2)
        else:
            x_toplot_list = whole_list[idx]
            plot_ridge_series(x_toplot_list, num_epoch_list, sample_size_list, yrange, titlestr, xstr, ystr, fig, gs, showYaxis)

def plot_ridge_series(x_toplot_list, num_epoch_list, sample_size_list, yrange, titlestr, xstr, ystr, fig, gs, showYaxis):
    cmap = plt.get_cmap('viridis', len(sample_size_list))
    colors = cmap.colors
    # prop_cycle = plt.rcParams['axes.prop_cycle']
    # colors = prop_cycle.by_key()['color']
    max_num_epoch = max(num_epoch_list)
    ymin = yrange[0]
    ymax = yrange[1]
    # print('min: %1.2f, max: %1.2f' % (ymin, ymax))

    num_t_plot = len(x_toplot_list)
    ax_objs = []
    # gs = gridspec.GridSpecFromSubplotSpec(num_t_plot,num_t_plot*2, subplot_spec=gs_outer)
    gs.update(hspace=-0.9)
    for idx, (x_toplot, total_sample) in enumerate(zip(x_toplot_list, sample_size_list)):  
        num_epoch = num_epoch_list[idx]
        # mu = np.apply_along_axis(stats.tmean, 0, x_toplot)
        mu = np.apply_along_axis(np.nanmedian, 0, x_toplot)
        lower = np.apply_along_axis(np.nanpercentile, 0, x_toplot, 25)
        upper = np.apply_along_axis(np.nanpercentile, 0, x_toplot, 75)

        # creating new axes object
        # ax_objs.append(fig.add_subplot(gs[(num_t_plot-idx):(num_t_plot*2-idx), idx:(num_t_plot+idx)]))
        ax_objs.append(fig.add_subplot(gs[(num_t_plot-idx-1):(num_t_plot-idx), (num_t_plot-idx):(num_t_plot*2-idx)]))

        # plotting the distribution
        ax_objs[-1].plot(range(1,num_epoch+1), mu, '-', color=colors[idx])
        ax_objs[-1].fill_between(range(1,num_epoch+1), lower, upper, color=colors[idx], alpha=0.2)
        ax_objs[-1].plot(range(max_num_epoch), np.ones(max_num_epoch)*max([0,ymin]), '--', color = [0.7,0.7,0.7])

        # setting uniform x and y lims
        ax_objs[-1].set_xlim(1,max_num_epoch+1)
        ax_objs[-1].set_ylim(ymin,ymax)

        # make background transparent
        rect = ax_objs[-1].patch
        rect.set_alpha(0)

        # remove borders, axis ticks, and labels
        ax_objs[-1].set_yticklabels([])
        ax_objs[-1].set_yticks(np.linspace(ymin, ymax, 5))

        # if idx == 0:
        #     ax_objs[-1].set_xlabel(xstr, fontsize=16,fontweight="bold")
        #     ax_objs[-1].set_xticks(range(1, max_num_epoch+1, 10))
        # else:
        #     ax_objs[-1].set_xticklabels([])

        spines = ["top","right","bottom"]
        for s in spines:
            ax_objs[-1].spines[s].set_visible(False)

        if showYaxis:
            ax_objs[-1].spines["left"].set_color([0.7,0.7,0.7])
        else:
            ax_objs[-1].set_yticks([])
            ax_objs[-1].spines["left"].set_visible(False)

        if ymin < 0:
            ax_objs[-1].spines["bottom"].set_position(('data', 0))
        ax_objs[-1].set_xticks(range(1, max_num_epoch+1, 10))

        if idx == 0:
            ax_objs[-1].set_xlabel(xstr, fontsize=16,fontweight="bold")

        # if idx == np.floor(len(x_toplot_list)/2):
        #     ax_objs[-1].set_ylabel(ystr, rotation=-60)
        #     ax_objs[-1].yaxis.set_label_position('right')

        ax_objs[-1].text(-0.02,max([0,ymin]),'s%d' % total_sample,fontweight="bold",fontsize=14,ha="right")
        ax_objs[-1].grid(False)

        if idx == num_t_plot-1:
            ax_objs[-1].text(np.floor(max_num_epoch/2), np.amax(upper), titlestr,fontweight="bold",fontsize=14, ha='center')
            # print(np.amax(upper))
    # ax_objs[-1].set_title(titlestr, fontweight="bold",fontsize=14)
    # plt.tight_layout()
    return ax_objs

def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()