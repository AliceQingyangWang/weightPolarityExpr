# import sys, os
# # insert at 1, 0 is the script path (or '' in REPL)
# sys.path.insert(1, os.path.join(os.path.dirname(os.getcwd()), 'XOR-HD'))

import os, argparse
import numpy as np
from itertools import cycle, islice
import pandas as pd
from helperfun import run_bash

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument(
                "--doRandInit",
                type=int, 
                default=2, 
                required=False,
                help="0: do random initilizatoin for the freezing networks, vanilla; 1: NOT do randInit, preTrained polarity; 2: do all three combinations!"
        )
        parser.add_argument(
                "--num_gpu", 
                type=int, 
                default=2, 
                required=False,
                help="Available GPU number"
        )
        parser.add_argument(
                "--num_run",
                type=int,
                default=1,
                required=False,
                help="total number of runs, will run the rest of the runs except the ones already existing"
        )
        parser.add_argument(
                "--baseFName",
                type=str,
                default='',
                required=False, 
                help = 'base name if you need any. Corresponds to the dataset'
        )
        parser.add_argument(
                "--resetType",
                type=str,
                default='posRand',
                required=False,
                help="all: all 4 run; posCon: sign(epsilon); posRand: sign(rand([0,epsilon])); zero: 0; flip: -weight"
        )
        parser.add_argument(
                "--preTrained",
                default='False',
                required=False,
                help="whether to load pretrained weights to AlexNet."
        )
        parser.add_argument(
                "--num_epoch",
                default=50,
                type = int,
                required=False,
                help="total number of epochs"
        )
        parser.add_argument(
                "--start_epoch",
                default=0,
                type = int,
                required=False,
                help="Which epoch to start training"
        )
        parser.add_argument(
                "--ckpt_freq",
                default=1,
                type=int,
                required=False,
                help='how often checkpoints should be saved. If saved for every epoch, the file system will explode. 0: only logs the best'
        )        
        parser.add_argument(
                "--no_freeze",
                default='False',
                required=False,
                help="if True, will not run freeze experiment"
        )    
        parser.add_argument(
                "--no_liquid",
                default='False',
                required=False,
                help="if True, will not run liquid experiment"
        )   
        parser.add_argument(
                "--doEarlyStopping",
                default='False',
                required=False,
                help="whether do early stopping for the runs"
        )
        args, _ = parser.parse_known_args()

        config = {}
        config['batch_size'] = 1000
        # will be iterating 10 epochs at a time over here. 
        config['num_epoch'] = args.num_epoch # 50 
        config['start_epoch'] = args.start_epoch
        config['ckpt_freq'] = args.ckpt_freq
        numRun = args.num_run
        run_list = list(range(numRun))
        # run_list = [i for i, e in enumerate(list(islice(cycle(range(2)), 20))) if e != 0]

        if args.baseFName == 'cifar10':
                sample_size_list = np.array([100, 250,500,750,1000,2500,5000,10000,25000,50000]) # cifar-10
                gpu_load_list = [.5,.5,.5,.5,.5,.5,1,1,1,1]# CIFAR-10
        elif args.baseFName == 'fashion_mnist':
                # note: putting 100 in the end shoud make it less load on the gpu
                sample_size_list = np.concatenate(([250,500,750], np.arange(1,7)*1000, [100])) # fashion-MNIST
                gpu_load_list = .5*np.ones(len(sample_size_list)) # FASHION-MNIST
        else:
                raise('dataset not supported!')
        # sample_size_list = np.array([100])
        # gpu_load_list = np.array([1])
        doWeightFreeze_list = [True, False]
        total_num_gpu = args.num_gpu
        gpuNum_list = list(islice(cycle(range(total_num_gpu)), len(run_list))) # this should work, and it's more equal on the two gpus. Aha, the load is different!! - okay will change that. Nicely done!

        config['doBatchLog'] = 'False'
        doEarlyStopping = False
        
        if args.resetType == 'all':
                resetType_list = ['posCon', 'posRand', 'zero', 'flip'] #
        else:
                resetType_list = [args.resetType]

        job_df = pd.DataFrame(columns=["job_identifier", "job_cmd", "path_to_output_file", "gpu_num", "gpu_load", "check_path"])
        job_accumulator = 0

        config['baseFName'] = args.baseFName

        if args.doRandInit == 0:
                config['doRandInit'] = True
                vanilla_trained_str = 'vanilla'
        elif args.doRandInit == 1:
                config['doRandInit'] = False
                vanilla_trained_str = 'pretrained'  
        elif args.doRandInit == 2:
                vanilla_trained_str = '' # place holder
                expr_list = ['vanilla', 'pretrained', 'finetune']
                config['doRandInit'] = True
        else:
                raise('doRandInit value not allowed!')

        if args.preTrained == 'True': # By design, this variable can override any option in doRandInit. 
                config['preTrained'] = 'True'
                vanilla_trained_str = 'finetune'
        if not vanilla_trained_str == '':
                expr_list = [vanilla_trained_str]

        for vanilla_trained_str in expr_list:
                if not job_accumulator == 0: # doRandInit==2 situation and not the first iteration. 
                        if vanilla_trained_str == 'pretrained':
                                config['doRandInit'] = False
                        elif vanilla_trained_str == 'finetune':
                                config['doRandInit'] = True
                                config['preTrained'] = 'True'
                        else:
                                raise('something is wrong when setting experiment params for %s' % vanilla_trained_str)

                for resetType in resetType_list:
                        config['resetType'] = resetType # 'posRand'

                        for (runIter, gpuNum) in zip(run_list, gpuNum_list):
                                config['runIter'] = runIter
                                # config['gpu_num'] = gpuNum # won't assign gpu num here, instead do it in the scheduler!!

                                for (sample_size, gpu_load) in zip(sample_size_list, gpu_load_list):
                                        config['sample_size'] = sample_size
                                        
                                        if not os.path.isdir(os.path.join(os.getcwd(), config['baseFName'], resetType)):
                                                os.makedirs(os.path.join(os.getcwd(), config['baseFName'], resetType))
                                        nohup_output_path = os.path.join(os.getcwd(), config['baseFName'], '%s/run%d.out' % (resetType, runIter))

                                        python_sciprt_path = os.path.join(os.getcwd(), 'batch_by_run.py') 
                                        # below is for checking if the job is done
                                        check_path = '%ss%d_%s_run%d' % (config['resetType'], config['sample_size'], vanilla_trained_str, runIter)

                                        job_cmd = 'nohup python %s ' % python_sciprt_path + '  '.join(
                                                ['--{}={}'.format(flag, value) for flag, value in config.items()]) + '> %s &' % nohup_output_path 

                                        path_to_output_file = os.path.join(os.getcwd(), config['baseFName'], '%s/run%d_bash.txt' % (resetType, runIter))                                        
                                        
                                        row_to_add = pd.DataFrame({"job_identifier": job_accumulator, 
                                                        "job_cmd": job_cmd, 
                                                        "path_to_output_file": path_to_output_file, 
                                                        "gpu_num": gpuNum, # this is only a suggested value
                                                        "gpu_load": gpu_load, 
                                                        "check_path": check_path}, index=[job_accumulator])
                                        # print(row_to_add)
                                        job_df = pd.concat([job_df, row_to_add], ignore_index=True)
                                        job_accumulator += 1

        print(job_df.head())

        df_path = os.path.join(args.baseFName, 'all.pkl')
        job_df.to_pickle(df_path)
        with open(os.path.join(args.baseFName, 'scheduler.txt'),'a+') as myoutput:
                python_sciprt_path = os.path.join(os.getcwd(), 'scheduler.py')
                run_bash('python %s --path %s --baseFName %s --total_num_gpu %d &' % (python_sciprt_path, df_path, args.baseFName, total_num_gpu), myoutput)

        # for thisGPU in set(gpuNum_list):
        #         job_subset = job_df.loc[job_df['gpu_num'] == thisGPU]
        #         df_path = os.path.join(args.baseFName, '%d.pkl' % thisGPU)
        #         job_subset.to_pickle(df_path)

        #         with open(os.path.join(args.baseFName, 'scheduler.txt'),'a+') as myoutput:
        #                 python_sciprt_path = os.path.join(os.getcwd(), 'scheduler.py')
        #                 run_bash('python %s --path %s --baseFName %s &' % (python_sciprt_path, df_path, args.baseFName), myoutput)