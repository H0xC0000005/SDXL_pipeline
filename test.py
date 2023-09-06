from util import PromptLoader
from main import SDXLInferencePipeline
import os

sp = SDXLInferencePipeline(verbose=True, refine=False,  device_id=2)
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
pl = PromptLoader("./play_temp.json")
img = sp.inference_with_prompt_loader(pl, batch_size=128, base_only=True, inference_steps=50, target_size=(768,768))
print(img[0])
dir_path = "./test_images/768x768/"
if not os.path.exists(dir_path):
    # If it doesn't exist, create it
    os.makedirs(dir_path)
pl.save_images(img, dir_path)

print(pl.get_filenames())



