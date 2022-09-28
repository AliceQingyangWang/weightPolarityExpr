# import numpy as np
import os, argparse, pickle
from sum_training_data import getMetrics_parallel

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
            "--resetType",
            type=str,
            default='all',
            required=False,
            help="all: all 4 run; posCon: sign(epsilon); posRand: sign(rand([0,epsilon])); zero: 0; flip: -weight"
    )
    args, _ = parser.parse_known_args()

    config = {}
    config['batch_size'] = 32

    doWeightFreeze_list = [True, False]
    sample_size_list = [  8,  20, 32, 40, 52, 60, 72, 80, 92, 100, 500] #, 1000, 5000
    num_epoch_list = [100] * len(sample_size_list)
    num_dim_list = [5]
    numVal = 1000

    hSize_list = list(range(2,30))  + [30,32,48,64] 

    numRun = args.num_run
    doBatchLog = False

    if args.resetType == 'all':
            resetType_list = ['posCon', 'posRand', 'zero', 'flip'] #
    else:
            resetType_list = [args.resetType]

    metrics = getMetrics_parallel(num_dim_list, resetType_list, hSize_list, num_epoch_list, sample_size_list, doWeightFreeze_list, numRun, args.baseFName)
 
    f_name = os.path.join(os.getcwd(), args.baseFName, 'epoch_acc_loss.pkl')
    with open(f_name, 'wb') as f:
        pickle.dump(metrics, f)