import pickle, os, argparse
from tf_training import set_gpu_config, eachIter_dualCond
from generators import tensorize_datasets
from helperfun import build_string_from_dict, get_dict_by_key

def get_config(args):
    config = {}
    config['resetType'] = args.resetType
    config['num_dim'] = args.num_dim
    config['batch_size'] = args.batch_size
    config['sample_size'] = args.sample_size

    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resetType", 
        type=str, 
        default=None, 
        required=True,
        help="Options: zero, flip, posRand, posCon. "
    )
    parser.add_argument(
        "--sample_size",
        type = int,
        required = True, 
        help = "Total sample given as trining set. "
    )
    parser.add_argument(
        "--runIter", 
        type=int,
        required=True,
        help="The current run. This is specific to batch_by_run.py"
    )
    parser.add_argument(
        "--num_epoch",
        type=int,
        required=True,
        help="This is associated with sample size. "
    )
    parser.add_argument(
        "--num_dim",
        type=int,
        default=5,
        required=False,
        help="Number of data dimension, 2 of which are relevant to XOR.",
    )
    parser.add_argument(
        "--gpu_num",
        type=int,
        default=0,
        required=True,
        help="which GPU you want to train the network on"
    )
    parser.add_argument(
        "--batch_size",
        type = int,
        default = 32,
        required=False,
        help = 'As name. '
    )
    parser.add_argument(
        "--doBatchLog",
        default=False,
        required=False,
        help="If logging batch info, it will take way longer time. Normally only do it for one run. Otherwise batched in epochs. Sufficient for most cases. "
    )
    parser.add_argument(
        "--doRandInit",
        default=False,
        required=False,
        help="triger where you can disallow resetting at the initialization step"
    )
    parser.add_argument(
        "--hSize",
        type=int,
        default=64,
        required=False,
        help="number of hidden units of the network"
    )
    parser.add_argument(
        "--baseFName",
        type=str,
        default='',
        required=False,
        help="the folder name outside of typeStr"
    )
    parser.add_argument(
        "--doEarlyStopping",
        default=False,
        help="whether do early stopping for the runs"
    )
    args, _ = parser.parse_known_args()
    if isinstance(args.doBatchLog, str):
        args.doBatchLog = args.doBatchLog=='True'
    if isinstance(args.doRandInit, str):
        args.doRandInit = args.doRandInit=='True'
    if isinstance(args.doEarlyStopping, str):
        args.doEarlyStopping = args.doEarlyStopping=='True'
    config = get_config(args)

    set_gpu_config()

    f_name = os.path.join(os.getcwd(), 'data', build_string_from_dict(get_dict_by_key(config, {'num_dim', 'sample_size'})) + '.pkl')
    with open(f_name, 'rb') as f:
        ds = pickle.load(f)
        # ALL CHECK
        # print(ds['dataset_train']['data'].shape)
        # print(ds['dataset_val']['data'].shape)
        # print(ds['dataset_train']['data'][:, :8])
        
    ds_train_threaded = []
    for tt in range(2): 
        ds_train = tensorize_datasets(ds['dataset_train'], config['batch_size'], True)
        # print(list(ds_train.as_numpy_iterator())[0])
        ds_train_threaded.append(ds_train)   
    ds_val = tensorize_datasets(ds['dataset_val'], config['batch_size'], False)

    total_sample = config['sample_size']
    runIter = args.runIter

    log_dir = os.path.join(os.getcwd(), args.baseFName, config['resetType'], 'logs', 's%de%dr%d' % (total_sample, args.num_epoch, args.hSize), "r%d_{typeStr}" % runIter) #runIter
    checkpoint_path = os.path.join(os.getcwd(), args.baseFName, config['resetType'], 'checkpoints', 's%de%dr%d' % (total_sample, args.num_epoch, args.hSize), "r%d_{typeStr}" % runIter)
    
    model_freeze, model_liquid = eachIter_dualCond(log_dir, checkpoint_path, ds_train_threaded, ds_val, config['num_dim'], args.num_epoch, config['resetType'], args.doBatchLog, args.gpu_num, args.doEarlyStopping, args.hSize, doRandInit = args.doRandInit)
    
    f_name = os.path.join(os.getcwd(), args.baseFName.split(os.path.sep)[0], 'status.txt')
    with open(f_name, 'a') as f:
        f.write('dim%d%ss%de%dr%drun%d\n' % (args.num_dim, config['resetType'], total_sample, args.num_epoch, args.hSize, runIter))
    print('expr_s%d_r%d_' % (total_sample, (runIter+1)) + 'is done')