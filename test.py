from util import PromptLoader
from main import SDXLInferencePipeline

sp = SDXLInferencePipeline(verbose=True)

img = sp(["an SR-71 spy plane", "a revolver"], inference_steps=50)
print(img)


