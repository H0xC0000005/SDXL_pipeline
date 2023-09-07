from util import PromptLoader
from main import SDXLInferencePipeline
import os
import time

start = time.time()
sp = SDXLInferencePipeline(verbose=True, refine=False,  device_id=3)
#
# # img = sp(["an SR-71 spy plane", "a cute girl, anime style", "a GPU cluster",
# #           "a researcher in front of some computer screens",
# #           "a street scene of Tokyo Station",
# #           "a computer science Phd student painting an acrylic painting"], inference_steps=30, base_only=True)
# prompts = ["a cute girl, anime style", "a GPU cluster"]
# negative_prompts = ['', '']
# img = sp(prompts, negative_prompts, inference_steps=50, base_only=False)
# print(img)
# print(img[0])
# print(type(img[0]))
# if not os.path.exists("./test_images/"):
#     # If it doesn't exist, create it
#     os.makedirs("./test_images/")
# for idx in range(len(img)):
#     img[idx].save(f"./test_images/test{idx}.png", "PNG")
pl = PromptLoader("./filtered_96_swapped.json")

dir_path = "./result_images/1024x1024/20230906_5000imgtest/"
if not os.path.exists(dir_path):
    # If it doesn't exist, create it
    os.makedirs(dir_path)
for img, filenames in sp.inference_with_prompt_loader(pl, batch_size=16, base_only=True, inference_steps=50, target_size=(1024,1024)):
    pl.save_images(img, dir_path, filenames)

print(pl.get_filenames())
end = time.time()

print(f">>> final elapsed time in sec: {start - end}")



