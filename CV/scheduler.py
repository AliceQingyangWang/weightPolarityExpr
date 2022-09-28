import argparse, time, os
from glob import glob
import pandas as pd
from helperfun import run_bash
import numpy as np
## will use a status.txt at the parent dir to record which experiments are done and use that to update job list. 

def update_job_list(job_df, job_running, job_completed, txt_path):
    # job_running should have #gpu streams
    # job_completed should be shared between the two

    jobs_to_check = job_df.loc[job_df['job_identifier'].isin(job_running)]
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            status_file_done_list = set(f.read().split('\n'))
    else:
        status_file_done_list = set([])
    # path_check = jobs_to_check["check_path"].apply(lambda x: print(x))

    finished = set(jobs_to_check['job_identifier'].loc[jobs_to_check["check_path"].isin(status_file_done_list)])
    # set(path_check.loc[lambda x: x].index.tolist())

    job_completed |= finished
    job_running -= finished
    # print(finished)
    # print(job_completed)
    # print(job_running)

    total_gpu_load = sum(job_df['gpu_load'].loc[job_df['job_identifier'].isin(job_running)].tolist())
    # print('Current gpu load on %d is %1.1f' % (job_df['gpu_num'].tolist()[0], total_gpu_load))

    return total_gpu_load

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", 
        type=str, 
        default=None, 
        required=True,
        help="Path where the job dataframe is stored at."
    )
    parser.add_argument(
        "--baseFName",
        type=str,
        default='',
        required=False, 
        help = 'base name if you need any'
    )
    parser.add_argument(
        "--total_num_gpu",
        type=int,
        default=2,
        required=False,
        help="number of available GPUs"
    )
    args, _ = parser.parse_known_args()
    job_df = pd.read_pickle(args.path)
    print(job_df)
    txt_path = os.path.join(args.baseFName, 'status.txt')

    ### initialize
    job_completed = set([])
    job_init = set(job_df.index.tolist())
    update_job_list(job_df, job_init, job_completed, txt_path); # will update job_completed, job_init is actually the whole list of jobs left to run

    job_running_list = []
    for _ in range(args.total_num_gpu):
        job_running_list.append(set([]))

    for idx, thisJob in job_df.iterrows(): 
        current_load = [update_job_list(job_df, job_running, job_completed, txt_path) + thisJob.gpu_load for job_running in job_running_list]
        # print(job_completed)
        # print(job_running_list)
        # print(current_load)   
        # print((np.array(current_load) > 1).all())
        while (np.array(current_load) > 1).all():
            time.sleep(10) 
            current_load = [update_job_list(job_df, job_running, job_completed, txt_path) + thisJob.gpu_load for job_running in job_running_list]
        
        # print(np.array(current_load) <= 1)
        if not thisJob['job_identifier'] in job_completed:
            gpu_num = next((i for i, j in enumerate(np.array(current_load) <= 1) if j), None)
            with open(thisJob.path_to_output_file,'a+') as myoutput:
                tmp = thisJob.job_cmd.split('>')
                tmp[0] += ' --{}={}'.format('gpu_num', gpu_num)
                thisJob.job_cmd = '>'.join(tmp)
                submit_cmd = "bash -c '%s'" % thisJob.job_cmd #ts --gpus 1 
                run_bash(submit_cmd, myoutput)
                # print(submit_cmd)
            job_running_list[gpu_num] |= set([thisJob.job_identifier])

