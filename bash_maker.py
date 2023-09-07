import json
import os

from util import PromptLoader


def make_job_bash(job_name: str, job_config_file_path: str | os.PathLike, gpu_id_list: list[int] | None = None, split_line: bool = False):
    if split_line:
        suffix - " \\ \n"
    else:
        suffix = ' '
    try:
        with open(job_config_file_path, "r") as read_file:
            data: dict = json.load(read_file)
    except OSError as e:
        print(e)
        raise NotImplementedError("currently only support JSON file, or you specify an invalid path")
    with open(f"{job_name}_{gpu_id_list}.sh", "w") as file:
        use_refiner = f"--use_refiner {suffix}" if data["use_refiner"] else ""
        base_model_path = f"--base_model_path {data['base_model_path']} {suffix}" if isinstance(data["base_model_path"], str) else ''
        refiner_model_path = f"--refiner_model_path {data['refiner_model_path']} {suffix}" if isinstance(data["refiner_model_path"], str) else ''
        cache_model = f"--cache_model {suffix}" if data["cache_model"] else ""
        device_id = data["device_id"]
        if gpu_id_list:
            device_id = list(set(device_id) & set(gpu_id_list))
            if not device_id:
                raise RuntimeError(f"specified gpu id list {gpu_id_list} has no intersection to "
                                   f"configured gpu list {device_id}")
        verbose = f"--verbose {suffix}" if data["verbose"] else ''

        prompt_loader_data_file_path = f'--prompt_loader_data_file_path {data["prompt_loader_data_file_path"]} {suffix}' if isinstance(data["prompt_loader_data_file_path"], str) else ''
        pl = PromptLoader(data["prompt_loader_data_file_path"])
        full_length = len(pl)
        partitions = len(device_id)
        size_of_partition = full_length // partitions
        cur_partition_start = 0

        batch_size = f'--batch_size {data["batch_size"]} {suffix}' if isinstance(data["batch_size"], int) else f"--batch_size 8 {suffix}"
        inference_steps = f'--inference_steps {data["inference_steps"]} {suffix}' if isinstance(data["inference_steps"], int) else f"--inference_steps 50 {suffix}"
        target_size = data["target_size"]
        target_size_str = f"--target_size {target_size[0]}:{target_size[1]} {suffix}"
        base_only_inference = f"--base_only_inference {suffix}" if not data["use_refiner"] else ''
        return_type = f'--return_type {data["return_type"]} {suffix}' if data["return_type"] else ''

        for gpu_id in device_id:
            # want to get consistent jobs to avoid potential problems
            cur_slice = (cur_partition_start, cur_partition_start + size_of_partition)
            responsible_slice_str = f"--responsible_slice {cur_slice[0]}:{cur_slice[1]} {suffix}"
            cur_partition_start += size_of_partition
            file.write(f"python3 {job_name}.py {suffix}"
                       f"{use_refiner}"
                       f"{base_model_path}"
                       f"{refiner_model_path}"
                       f"{cache_model}"
                       f"--device_id {gpu_id} {suffix}"
                       f"{verbose}"
                       f"{prompt_loader_data_file_path}"
                       f"{responsible_slice_str}"
                       f"{batch_size}"
                       f"{inference_steps}"
                       f"{target_size_str}"
                       f"{base_only_inference}"
                       f"{return_type} & \n")


if __name__ == "__main__":
    # def make_job_bash(job_name: str, job_config_file_path: str | os.PathLike, gpu_id_list: list[int] | None = None, ):
    make_job_bash("main", "job_config.json")
