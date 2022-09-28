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
                default=1, # do rand initialization by default
                required=False,
                help="1: do random initilizatoin for the freezing networks; 0: NOT do randInit"
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
                help = 'base name if you need any'
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
        total_num_gpu = args.num_gpu
        numVal = 1000

        hSize_list = list(range(2,30))  + [30,32,48,64] 
        gpuNum_list = list(islice(cycle(range(total_num_gpu)), len(hSize_list)))
        gpu_load_list = np.ones(len(hSize_list))*0.05 # 20 jobs/gpu

        numRun = args.num_run
        doBatchLog = False
        doEarlyStopping = False
        if args.doRandInit == 1:
                doRandInit = True
        else:
                doRandInit = False
        
        if args.resetType == 'all':
                resetType_list = ['posCon', 'posRand', 'zero', 'flip'] #
        else:
                resetType_list = [args.resetType]

        job_df = pd.DataFrame(columns=["job_identifier", "job_cmd", "path_to_output_file", "gpu_num", "gpu_load", "check_path"])
        job_accumulator = 0

        for runIter in range(numRun):
                config['runIter'] = runIter

                for num_dim in num_dim_list:
                        config['num_dim'] = num_dim
                        config['baseFName'] = os.path.join(args.baseFName, 'dim%d' % config['num_dim'])

                        for resetType in resetType_list:
                                config['resetType'] = resetType # 'posRand'

                                for (hSize, gpuNum, gpu_load) in zip(hSize_list, gpuNum_list, gpu_load_list): 
                                        config['hSize'] = hSize
                                        config['gpu_num'] = gpuNum# gpu_num_list[idx]

                                        for (total_sample, num_epoch) in zip(sample_size_list, num_epoch_list):
                                                config['sample_size'] = total_sample
                                                config['num_epoch'] = num_epoch
                                                config['doRandInit'] = doRandInit

                                                if not os.path.isdir(os.path.join(os.getcwd(), config['baseFName'], resetType)):
                                                        os.makedirs(os.path.join(os.getcwd(), config['baseFName'], resetType))
                                                nohup_output_path = os.path.join(os.getcwd(), config['baseFName'], '%s/s%dr%d.out' % (resetType, total_sample, hSize))

                                                python_sciprt_path = os.path.join(os.getcwd(), 'batch_by_run.py') 
                                                # below is for checking if the job is done
                                                check_path = 'dim%d%ss%de%dr%drun%d' % (num_dim, config['resetType'], total_sample, num_epoch, hSize, runIter)

                                                job_cmd = 'nohup python %s ' % python_sciprt_path + '  '.join(
                                                        ['--{}={}'.format(flag, value) for flag, value in config.items()]) + '> %s &' % nohup_output_path 

                                                path_to_output_file = os.path.join(os.getcwd(), config['baseFName'], '%s/s%dr%d_bash.txt' % (resetType, total_sample, hSize))                                        
                                                
                                                row_to_add = pd.DataFrame({"job_identifier": job_accumulator, 
                                                                "job_cmd": job_cmd, 
                                                                "path_to_output_file": path_to_output_file, 
                                                                "gpu_num": gpuNum, 
                                                                "gpu_load": gpu_load, 
                                                                "check_path": check_path}, index=[job_accumulator])

                                                job_df = pd.concat([job_df, row_to_add], ignore_index=True)
                                                job_accumulator += 1

        print(job_df.head())

        for thisGPU in set(gpuNum_list):
                job_subset = job_df.loc[job_df['gpu_num'] == thisGPU]
                df_path = os.path.join(args.baseFName, '%d.pkl' % thisGPU)
                job_subset.to_pickle(df_path)

                with open(os.path.join(args.baseFName, 'scheduler.txt'),'a+') as myoutput:
                        python_sciprt_path = os.path.join(os.getcwd(), 'scheduler.py')
                        run_bash('python %s --path %s --baseFName %s &' % (python_sciprt_path, df_path, args.baseFName), myoutput)