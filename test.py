from util import PromptLoader
from main import SDXLInferencePipeline

sp = SDXLInferencePipeline(verbose=False)

img = sp(["an SR-71 spy plane", "a revolver"], inference_steps=50, base_only=True)
print(img)
img[0].save("test.png", "PNG")
img[1].save("test2.png", "PNG")

