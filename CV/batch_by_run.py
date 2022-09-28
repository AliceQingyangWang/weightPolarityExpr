# import sys, os
# # insert at 1, 0 is the script path (or '' in REPL)
# sys.path.insert(1, os.path.join(os.path.dirname(os.getcwd()), 'XOR-HD'))
import pickle, os, argparse
from tf_training import set_gpu_config, eachIter_dualCond
from helperfun import build_string_from_dict, get_dict_by_key
from data_cv import data_cv

def get_config(args):
    config = {}
    config['resetType'] = args.resetType
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
        "--runIter", 
        type=int,
        required=True,
        help="The current run. This is specific to batch_by_run.py"
    )
    parser.add_argument(
        "--sample_size",
        type = int,
        required = True, 
        help = "Total sample given as trining set. "
    )
    parser.add_argument(
        "--num_epoch",
        type=int,
        required=True,
        help=""
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
        default='False',
        required=False,
        help="If logging batch info, it will take way longer time. Normally only do it for one run. Otherwise batched in epochs. Sufficient for most cases. "
    )
    parser.add_argument(
        "--doRandInit",
        default='False',
        required=False,
        help="triger where you can disallow resetting at the initialization step"
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
        default='False',
        required=False,
        help="whether do early stopping for the runs"
    )
    parser.add_argument(
        "--preTrained",
        default='False',
        required=False,
        help="whether to load pretrained weights to AlexNet."
    )
    parser.add_argument(
        "--start_epoch",
        default=0,
        type=int,
        required=False,
        help="specify which epoch to start training at. This is mainly for picking up training and used for only once."
    )
    parser.add_argument(
        "--ckpt_freq",
        default=1,
        type=int,
        required=False,
        help='how often checkpoints should be saved. If saved for every epoch, the file system will explode. '
    )
    args, _ = parser.parse_known_args()
    if isinstance(args.doBatchLog, str):
        args.doBatchLog = args.doBatchLog=='True'
    if isinstance(args.doRandInit, str):
        args.doRandInit = args.doRandInit=='True'
    if isinstance(args.doEarlyStopping, str):
        args.doEarlyStopping = args.doEarlyStopping=='True'
    if isinstance(args.preTrained, str):
        args.preTrained = args.preTrained=='True'
    config = get_config(args)

    set_gpu_config()

    # Load data here
    fashion_mnist = data_cv(args.baseFName, config['sample_size'])
    ds_train_threaded = [fashion_mnist.get_train(), fashion_mnist.get_train()]
    ds_val = fashion_mnist.get_test()
    inputSize = 227
    for ii in range(len(ds_train_threaded)):
        ds_train_threaded[ii] = fashion_mnist.prepare_ds(ds_train_threaded[ii], inputSize, config['batch_size'], True)
    ds_val = fashion_mnist.prepare_ds(ds_val, inputSize, config['batch_size'], False)

    runIter = args.runIter

    if args.doRandInit:
        vanilla_trained_str = 'vanilla'
    else:
        vanilla_trained_str = 'pretrained'
    if args.preTrained:
        vanilla_trained_str = 'finetune'
    # log_dir = os.path.join(os.getcwd(), args.baseFName, config['resetType'], 'logs', 's%de%d_%s' % (config['sample_size'], args.num_epoch, vanilla_trained_str), "r%d_{typeStr}" % runIter) 
    # checkpoint_path = os.path.join(os.getcwd(), args.baseFName, config['resetType'], 'checkpoints', 's%de%d_%s' % (config['sample_size'], args.num_epoch, vanilla_trained_str), "r%d_{typeStr}" % runIter)
    log_dir = os.path.join(os.getcwd(), args.baseFName, config['resetType'], 'logs', 's%de%d_%s' % (config['sample_size'], 50, vanilla_trained_str), "r%d_{typeStr}" % runIter) 
    checkpoint_path = os.path.join(os.getcwd(), args.baseFName, config['resetType'], 'checkpoints', 's%de%d_%s' % (config['sample_size'], 50, vanilla_trained_str), "r%d_{typeStr}" % runIter)
    
    model_config = {}
    model_config['mType'] = 'AlexNet'
    model_config['pre_trained'] = args.preTrained
    if ~args.preTrained:#hard-coded for now.... For now it should never be true because you are not directly transfer traning, you are only using the polarities!!
        model_config['channels'] = 3
        model_config['image_width'] = 227
        model_config['image_height'] = 227
        model_config['num_classes'] = 10
    
    model_freeze, model_liquid = eachIter_dualCond(log_dir, checkpoint_path, model_config, ds_train_threaded, ds_val, args.num_epoch, config['resetType'], args.doBatchLog, args.gpu_num, args.doEarlyStopping, doRandInit = args.doRandInit, start_epoch = args.start_epoch, ckpt_freq = args.ckpt_freq)
    
    f_name = os.path.join(os.getcwd(), args.baseFName.split(os.path.sep)[0], 'status.txt')
    with open(f_name, 'a') as f:
        f.write('%ss%d_%s_run%d\n' % (config['resetType'], config['sample_size'], vanilla_trained_str, runIter))
    print('expr_run%d_' % ((runIter+1)) + 'is done')