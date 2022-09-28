# import numpy as np
import os, argparse, pickle
import numpy as np
from main import getMetrics_parallel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_run",
        type=int,
        default=1,
        required=False,
        help="total number of runs, will only collect the first few runs"
    )
    parser.add_argument(
        "--baseFName",
        type=str,
        default = '',
        required = False,
        help = "the base folder name of the experiments"
    )
    parser.add_argument(
        "--saveFName",
        type=str,
        default = '',
        required = False,
        help = "the base folder name of the experiments"
    )
    parser.add_argument(
        "--resetType",
        type=str,
        default='all',
        required=False,
        help="all: all 4 run; posCon: sign(epsilon); posRand: sign(rand([0,epsilon])); zero: 0; flip: -weight"
    )
    parser.add_argument(
        "--numEpoch",
        type=int,
        default=50,
        required=False,
        help="num of epochs to collect data from."
    )
    args, _ = parser.parse_known_args()

    config = {}
    config['batch_size'] = 1000

    doWeightFreeze_list = [True, False]
    dataset_type = args.baseFName.split(os.sep)[-1]
    if dataset_type == 'cifar10':
        sample_size_list = np.array([100,250,500,750,1000,2500,5000,10000,25000,50000]) # cifar10
    elif dataset_type == 'fashion_mnist':
        sample_size_list = np.concatenate(([100,250,500,750], np.arange(1,7)*1000)) # fashion-MNIST: 
    else:
        raise('Non-compatible dataset type!!')
    num_epoch_list = [args.numEpoch] * len(sample_size_list)
    networkType_list = ['pretrained', 'vanilla', 'finetune']

    numRun = args.num_run

    if args.resetType == 'all':
            resetType_list = ['posCon', 'posRand', 'zero', 'flip'] #
    else:
            resetType_list = [args.resetType]

    metrics = getMetrics_parallel(resetType_list, networkType_list, num_epoch_list, sample_size_list, doWeightFreeze_list, numRun, args.baseFName)

    f_name = os.path.join(os.getcwd(), args.saveFName, 'epoch_acc_loss.pkl')
    with open(f_name, 'wb') as f:
        pickle.dump(metrics, f)

    # def get_converge_epoch(x):
    #     epoch_num = np.squeeze(np.where(x == np.amin(x)))
    #     if not len(epoch_num.shape)==0:
    #         epoch_num = epoch_num[0]
    #     return epoch_num

    # # getting convergence at each individual run
    # acc_at_conver_whole = {}
    # epoch_at_conver_whole = {}
    # for num_dim in num_dim_list:
    #     acc_at_conver = {}
    #     epoch_at_conver = {}

    #     for doWeightFreeze in doWeightFreeze_list:
    #         if doWeightFreeze:
    #             typeStr = 'freeze'
    #         else:
    #             typeStr = 'liquid'

    #         this_epoch_conver = np.empty(shape=(numRun, len(hSize_list)), dtype = np.float)
    #         this_acc_conver = np.empty(shape=(numRun, len(hSize_list)), dtype = np.float)
    #         for idx, total_sample in enumerate(hSize_list):
    #             x = val_loss_whole[folderName][typeStr][total_sample]
    #             z = val_acc_whole[folderName][typeStr][total_sample]

    #             for rowNum in range(x.shape[0]): #
    #                 thisR = x[rowNum, :]
    #                 epochNum = get_converge_epoch(thisR)
                    
    #                 acc_convergence = z[rowNum, epochNum]
    #                 this_epoch_conver[rowNum, idx] = epochNum
    #                 this_acc_conver[rowNum, idx] = acc_convergence

    #         epoch_at_conver[typeStr] = this_epoch_conver
    #         acc_at_conver[typeStr] = this_acc_conver
    #     acc_at_conver_whole[num_dim] = acc_at_conver
    #     epoch_at_conver_whole[num_dim] = epoch_at_conver

    # f_name = os.path.join(os.getcwd(), baseFName, 'epoch_acc_converge.pkl')
    # with open(f_name, 'wb') as f:
    #     ds = pickle.dump([acc_at_conver_whole,
    #                     epoch_at_conver_whole], f) #,acc_at_conver_collective, epoch_at_conver_collective