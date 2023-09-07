python3 test_job_bash.py \ 
 --base_model_path ./models/SDXL_base/ \ 
--refiner_model_path ./models/SDXL_refiner \ 
--device_id 0 \ 
 --prompt_loader_data_file_path ./play.json \ 
--responsible_slice 0:24 \ 
--batch_size 16 \ 
--inference_steps 50 \ 
--target_size 1024:1024 \ 
--base_only_inference \ 
--return_type pil \ 
& 
python3 test_job_bash.py \ 
 --base_model_path ./models/SDXL_base/ \ 
--refiner_model_path ./models/SDXL_refiner \ 
--device_id 1 \ 
 --prompt_loader_data_file_path ./play.json \ 
--responsible_slice 24:48 \ 
--batch_size 16 \ 
--inference_steps 50 \ 
--target_size 1024:1024 \ 
--base_only_inference \ 
--return_type pil \ 
& 