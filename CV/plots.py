import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from numpy import matlib
from scipy.stats import sem

def plot_median_plus_example(x_mat, x_coord, ylim, xlim, label, doExample, thisAx, color=[1,1,0], lineType='-'):
    # rows are repeats, columns are time stamps
    rect = thisAx.patch
    rect.set_alpha(0)

    mu = np.apply_along_axis(np.nanmedian, 0, x_mat)
    lower = np.apply_along_axis(np.nanpercentile, 0, x_mat, 25)
    upper = np.apply_along_axis(np.nanpercentile, 0, x_mat, 75)

    print(mu)
    thisAx.plot(x_coord, mu, lineType, color=color, label=label)
    thisAx.fill_between(x_coord, lower, upper, color=color, alpha=0.2)

    if doExample:
        # thisAx.plot(x_coord, np.ones(x_coord.shape[0]),'--', color=[1,1,1])
        thisAx.plot(x_coord, np.transpose(x_mat[np.random.randint(0, x_mat.shape[0]),:]), '--', color=color)
    
    if not ylim == 0:
        thisAx.set_ylim(ylim)
    thisAx.set_xlim(xlim)
    return thisAx.get_ylim()

from scipy.stats import mannwhitneyu
def plot_diff_plus_mannwhitneyu(x_mat, y_mat, x_coord, ylim, xlim, label, thisAx, color=[1,1,0], lineType='-', linewidth=4):
    # rows are repeats, columns are time stamps
    # Mann Whitney U tests for 'less': x<y
    rect = thisAx.patch
    rect.set_alpha(0)

    x_mu = np.apply_along_axis(np.nanmedian, 0, x_mat)
    y_mu = np.apply_along_axis(np.nanmedian, 0, y_mat)
    diff = y_mu - x_mu

    print(x_mu)
    print(y_mu)
    thisAx.plot(x_coord, diff, lineType, color=color, label=label, linewidth=linewidth)
    test_stat = mannwhitneyu(x_mat, y_mat, axis = 0, alternative = 'less', nan_policy = 'omit') #
    print(test_stat)
    for (x, thisDiff, thisStat) in zip(x_coord, diff, test_stat.pvalue):
        if thisStat <= 0.05:
            # thisAx.text(x*.8, thisDiff, '%1.2f' % thisStat)
            thisAx.text(x*.9, thisDiff, '*', fontsize = 20, color = color)
    
    if not ylim == 0:
        thisAx.set_ylim(ylim)
    thisAx.set_xlim(xlim)
    return thisAx.get_ylim()
    
def plot_sem(x_mat, x_coord, ylim, xlim, label, thisAx, color=[1,1,0], lineType='-'):
    # rows are repeats, columns are time stamps
    rect = thisAx.patch
    rect.set_alpha(0)

    mu = np.apply_along_axis(np.nanmean, 0, x_mat)
    sem_val = np.apply_along_axis(sem, 0, x_mat, nan_policy='omit')
    lower = mu-sem_val
    upper = mu+sem_val

    print(mu)
    thisAx.plot(x_coord, mu, lineType, color=color, label=label)
    thisAx.fill_between(x_coord, lower, upper, color=color, alpha=0.2)
    
    if not ylim == 0:
        thisAx.set_ylim(ylim)
    thisAx.set_xlim(xlim)
    return thisAx.get_ylim()

def get_format(flag, value): #TODO: not robust at all....
    if isinstance(value, str):
        if '@' in value:
            return '{} in {}'.format(flag, value)
    return '{}=={}'.format(flag, value)

def dict_to_query(dict_in):
    # return ' & '.join(['{}=={}'.format(flag, value) for flag, value in dict_in.items()])
    return ' & '.join([get_format(flag, value) for flag, value in dict_in.items()])

def plot_training_curves(metrics, col_name, hiddenSize_list, page_dict, doExample, fig):
    gs = gridspec.GridSpec(5,6)
    loc_gridx, loc_gridy = np.meshgrid(np.arange(5), np.arange(6))
    loc_gridx = np.transpose(loc_gridx).reshape([-1])
    loc_gridy = np.transpose(loc_gridy).reshape([-1])

    if 'acc' in col_name:
        ylim = [0,1]
    else:
        ylim = 0
    x_coord = np.arange(page_dict['num_epoch'])
    xlim = [0,page_dict['num_epoch']]

    for idx, hiddenSize in enumerate(hiddenSize_list):
        thisAx = fig.add_subplot(gs[(loc_gridx[idx]):(loc_gridx[idx]+1), (loc_gridy[idx]):(loc_gridy[idx]+1)])
        page_dict['hiddenSize'] = hiddenSize

        page_dict['typeStr'] = '\'freeze\''
        plot_median_plus_example(metrics.query(dict_to_query(page_dict))[col_name].tolist()[0], x_coord, ylim, xlim, 'freeze', doExample, plt.gca())
        page_dict['typeStr'] = '\'liquid\''
        ylim_real = plot_median_plus_example(metrics.query(dict_to_query(page_dict))[col_name].tolist()[0], x_coord, ylim, xlim, 'liquid', doExample, plt.gca(), color=[255/255,140/255,0])
        thisAx.text(page_dict['num_epoch']/2, ylim_real[1]*1.06, 'widht = %d hidden units' % hiddenSize,ha='center')

    del page_dict['hiddenSize']
    del page_dict['typeStr']

def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def sum_plotting(whole_list, num_epoch_list, sample_size_list, resetType_list, xstr, ystr, fig, showYaxis, doTitle, yLabel='', yLabelPos=[0,.5], fontSize=16):
    num_t_plot = len(sample_size_list)
   
    # fig = plt.figure(figsize=(num_t_plot*3,num_t_plot*2))
    # rect = fig.patch
    # rect.set_facecolor([0.6,0.6,0.6])

    if len(resetType_list)>1:
        # gs_w = 0.28
        # heading_increment = (1-gs_w)/(len(resetType_list)-1)
        # gs_divide_lines = np.concatenate([[0], np.cumsum(np.ones(len(resetType_list)-1)*heading_increment), [1]])
        # will have at most two columns. can have however many rows....
        num_row = int(np.ceil(len(resetType_list)/2))
        gs_w = 0.5
        vert_divide_lines = np.array([0, 0.5])
        horz_divide_lines = np.array([0, 0.45, 0.55, 1])
        vert_indices = vert_divide_lines[matlib.repmat(np.array([[0, 1]]), 1, num_row)]
        horz_indices = horz_divide_lines[np.repeat(np.array([[0,2], [1,3]]), (np.ones(2)*2).astype(int), axis=1)]
    else:
        # gs_w = 1
        # heading_increment = 0
        # gs_divide_lines = [0,1]
        gs_w = 1
        vert_indices = np.array([[0],[1]])
        horz_indices = np.array([[0],[1]])
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
        if doTitle:
            titlestr = folderName
        else:
            titlestr = ''
        gs = gridspec.GridSpec(num_t_plot,num_t_plot*2)

        # gs.update(left = gs_divide_lines[idx], right = gs_divide_lines[idx]+gs_w, bottom=0, top=1)
        gs.update(left = vert_indices[0, idx], right = vert_indices[0, idx]+gs_w, bottom = horz_indices[0, idx], top = horz_indices[1, idx])
        if isinstance(whole_list, dict):
            x_toplot_list = whole_list['freeze'][idx]
            ax_objs = plot_ridge_series(x_toplot_list, num_epoch_list, sample_size_list, yrange, titlestr, xstr, ystr, fig, gs, showYaxis, yLabel, yLabelPos, fontSize)
            
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
            plot_ridge_series(x_toplot_list, num_epoch_list, sample_size_list, yrange, titlestr, xstr, ystr, fig, gs, showYaxis, yLabel, yLabelPos, fontSize)

def plot_ridge_series(x_toplot_list, num_epoch_list, sample_size_list, yrange, titlestr, xstr, ystr, fig, gs, showYaxis, yLabel='', yLabelPos=[0,.5], fontSize=14):
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
        # ax_objs[-1].set_xticks(range(1, max_num_epoch+1, 10))
        ax_objs[-1].set_xticks([])
        # ax_objs[-1].tick_params(axis='x', labelsize = fontSize)

        if idx == 0:
            # ax_objs[-1].set_xlabel(xstr, fontsize=16,fontweight="bold")
            ax_objs[-1].set_xlabel(xstr, fontsize=fontSize)
        if idx==(num_t_plot-1):
            ax_objs[-1].set_ylabel(yLabel, fontsize=fontSize)
            if not ((yLabelPos[0]==0) & (yLabelPos[1]==.5)):
                ax_objs[-1].yaxis.set_label_coords(yLabelPos[0], yLabelPos[1])

        # if idx == np.floor(len(x_toplot_list)/2):
        #     ax_objs[-1].set_ylabel(ystr, rotation=-60)
        #     ax_objs[-1].yaxis.set_label_position('right')

        ax_objs[-1].text(-0.02,max([0,ymin]),'s%d' % total_sample,fontweight="bold",fontsize=fontSize,ha="right")
        ax_objs[-1].grid(False)

        if idx == num_t_plot-1:
            ax_objs[-1].text(np.floor(max_num_epoch/2), np.amax(upper), titlestr,fontweight="bold",fontsize=fontSize, ha='center')
            # print(np.amax(upper))
    # ax_objs[-1].set_title(titlestr, fontweight="bold",fontsize=14)
    # plt.tight_layout()
    return ax_objs
