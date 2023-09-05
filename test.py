from util import PromptLoader
from main import SDXLInferencePipeline
import os

sp = SDXLInferencePipeline(verbose=False, refine=False)
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
pl = PromptLoader("./play.json")
img = sp.inference_with_prompt_loader(pl, batch_size=3, base_only=True)

if not os.path.exists("./test_images/"):
    # If it doesn't exist, create it
    os.makedirs("./test_images/")
for idx in range(len(img)):
    img[idx].save(f"./test_images/test{idx}.png", "PNG")



