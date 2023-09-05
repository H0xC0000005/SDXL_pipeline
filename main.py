import argparse

import transformers
import accelerate
import diffusers
import torch
import torchvision
from typing import Optional, Dict, List, Tuple
import safetensors
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionPipeline


class SDXLInferencePipeline:
    BASE_ONLINE_PATH = "stabilityai/stable-diffusion-xl-base-1.0"
    REFINER_ONLINE_PATH = "stabilityai/stable-diffusion-xl-refiner-1.0"

    def __init__(self, base_model_path: str = "./model_cache/SDXL_base",
                 refiner_model_path: str = "./model_cache/SDXL_refiner",
                 cache_model: bool = True, refine: bool = True,
                 verbose: bool = False):
        if base_model_path:
            try:
                self.base = StableDiffusionXLPipeline.from_pretrained(base_model_path,
                                                                      torch_dtype=torch.float16,
                                                                      variant="fp16",
                                                                      use_safetensors=True)
            except Exception as e:
                print(type(e))
                print(e)
                print(f"trying to load base model from online source and cache this model if possible")
                try:
                    self.base = StableDiffusionXLPipeline.from_pretrained(self.BASE_ONLINE_PATH,
                                                                          torch_dtype=torch.float16,
                                                                          variant="fp16",
                                                                          use_safetensors=True)
                except Exception as e2:
                    print(f"Failed to load from online fallback path with exception: {e2}")
                    raise e2
                if cache_model:
                    self.base.save_pretrained(base_model_path)
                    print(f"base model is cached at {base_model_path}.")
            try:
                # need to integrate with accelerate here
                self.base.to("cuda")
            except Exception as e:
                print(type(e))
                raise e
            if refine:
                """
                refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=pipe.text_encoder_2,
    vae=pipe.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
                """
                try:
                    self.refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                        refiner_model_path,
                        torch_dtype=torch.float16, 
                        variant="fp16", use_safetensors=True
                    )
                except Exception as e:
                    print(type(e))
                    print(e)
                    print(f"trying to load refine model from online source and cache this model if possible")
                    try:
                        self.refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                            "stabilityai/stable-diffusion-xl-refiner-1.0",
                            torch_dtype=torch.float16, variant="fp16", use_safetensors=True
                        )
                    except Exception as e2:
                        print(f"Failed to load from online fallback path with exception: {e2}")
                        raise e2
                    if cache_model:
                        self.refiner.save_pretrained(refiner_model_path)
                        print(f"refiner model is cached at {refiner_model_path}.")
                try:
                    self.refiner.to("cuda")
                except Exception as e:
                    print(type(e))
                    raise(e)

            else:
                self.refiner = None
            if verbose:
                print(self.base)
                print(self.refiner)

    def __call__(self, prompts: List[str], inference_steps: int = 50, high_noise_frac: int = 0.8,
                 target_size: Tuple[int, int] = (512, 512),
                 base_only: bool = False,
                 return_type: str = "pil"):
        assert return_type.lower() in ("pil", "latent"), f"return type in __call__ should be one of: pil, latent"
        if not base_only and self.refiner is not None:
            if self.refiner is None:
                print(f"WARNING: using a base-only SDXL pipeline with refiner flag on (i.e. base_only = False)")
                raise ValueError("using a base-only SDXL pipeline with refiner flag on (i.e. base_only = False)")
            return_type = "latent"
            base_images = self.base(prompt=prompts, output_type=return_type, num_inference_steps=inference_steps,
                                    ).images
            refined_images = self.refiner(prompt=prompts, num_inference_steps=inference_steps, image=base_images,
                                          target_size=target_size).images
            return refined_images
        else:
            return self.base(prompt=prompts, output_type=return_type, num_inference_steps=inference_steps,
                             target_size=target_size).images

    def postprocessing(self, returned_images):
        pass


def parse_arg():
    parser = argparse.ArgumentParser(prog="Arguments to run SDXL inference pipeline",
                                     description="Arguments to run SDXL inference pipeline")


if __name__ == "__main__":
    sdxl_pipeline = SDXLInferencePipeline()
