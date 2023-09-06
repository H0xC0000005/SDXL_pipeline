import time

def make_job_bash(job_name: str, gpu_id_list: list[int]):
    with open(f"{job_name}_{gpu_id_list}.sh", "+w"): 
        for gpu_id in gpu_id_list:
        
    

