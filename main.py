import argparse

import transformers
import accelerate
import diffusers
import torch
import torchvision
from typing import Optional, Dict, List, Tuple
import safetensors
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionPipeline



from typing import Optional, Dict, List, Tuple, Iterable, Sized
import safetensors
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionPipeline
from util import PromptLoader


class SDXLInferencePipeline:
    BASE_ONLINE_PATH = "stabilityai/stable-diffusion-xl-base-1.0"
    REFINER_ONLINE_PATH = "stabilityai/stable-diffusion-xl-refiner-1.0"

    def __init__(self, base_model_path: str = "./model_cache/SDXL_base",
                 refiner_model_path: str = "./model_cache/SDXL_refiner",
                 cache_model: bool = True, refine: bool = True,
                 verbose: bool = False,
                 device_id: int = 0):
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
        self.base.enable_vae_slicing()
        self.base.enable_vae_tiling()
        try:
            # need to integrate with accelerate here
            self.base.to(f"cuda:{device_id}")
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
            "stabilityai/stable-diffusion-xl-refiner-1.0"
            """
            try:
                self.refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    refiner_model_path,
                    text_encoder_2=self.base.text_encoder_2,
                    vae=self.base.vae,
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
                        text_encoder_2=self.base.text_encoder_2,
                        vae=self.base.vae,
                        torch_dtype=torch.float16,
                        variant="fp16", use_safetensors=True
                    )
                except Exception as e2:
                    print(f"Failed to load from online fallback path with exception: {e2}")
                    raise e2
                if cache_model:
                    self.refiner.save_pretrained(refiner_model_path)
                    print(f"refiner model is cached at {refiner_model_path}.")
            self.refiner.enable_vae_slicing()
            self.refiner.enable_vae_tiling()
            try:
                self.refiner.to("cuda")
            except Exception as e:
                print(type(e))
                raise (e)

        else:
            self.refiner = None
        if verbose:
            print(self.base)
            print(self.refiner)

    @classmethod
    def from_namespace(cls, ns: argparse.Namespace):
        pass
        """
            def __init__(self, base_model_path: str = "./model_cache/SDXL_base",
                 refiner_model_path: str = "./model_cache/SDXL_refiner",
                 cache_model: bool = True, refine: bool = True,
                 verbose: bool = False):"""
        base_model_path = ns.base_model_path
        refiner_model_path = ns.refiner_model_path
        cache_model = ns.cache_model
        refine = ns.use_refiner
        verbose = False
        return cls(base_model_path,
                   refiner_model_path,
                   cache_model,
                   refine,
                   verbose)

    def __call__(self, prompts, negative_prompts = None,
                 inference_steps: int = 50,
                 target_size: Tuple[int, int] = (512, 512),
                 base_only: bool = False,
                 return_type: str = "pil"):
        assert return_type.lower() in ("pil", "latent"), f"return type in __call__ should be one of: pil, latent"
        if negative_prompts is None:
            negative_prompts = [''] * len(prompts)
        else:
            assert len(negative_prompts) == len(prompts), (f"in SDXL inference, get prompt and negative prompt with"
                                                           f" different size (i.e. inconsistent batch)")
        if not base_only and self.refiner is not None:
            if self.refiner is None:
                print(f"WARNING: using a base-only SDXL pipeline with refiner flag on (i.e. base_only = False)")
                raise ValueError("using a base-only SDXL pipeline with refiner flag on (i.e. base_only = False)")
            return_type = "latent"
            base_images = self.base(prompt=prompts,
                                    negative_prompt=negative_prompts,
                                    output_type=return_type,
                                    num_inference_steps=inference_steps,
                                    original_size=target_size,
                                    target_size=target_size
                                    ).images
            refined_images = self.refiner(prompt=prompts,
                                          negative_prompt=negative_prompts,
                                          num_inference_steps=inference_steps,
                                          image=base_images,
                                          original_size=target_size,
                                          target_size=target_size
                                          ).images
            return refined_images
        else:
            return self.base(prompt=prompts,
                             negative_prompt=negative_prompts,
                             output_type=return_type,
                             num_inference_steps=inference_steps,
                             original_size=target_size,
                             target_size=target_size
                             ).images


    def inference_with_prompt_loader(self,
                                     loader: PromptLoader,
                                     batch_size: int = 4,
                                     inference_steps: int = 50,
                                     target_size: Tuple[int, int] = (512, 512),
                                     base_only: bool = False,
                                     return_type: str = "pil"
                                     ):
        print(f">>> inference with a prompt loader with batch size {batch_size} and target output size {target_size}.")
        all_images = []
        while True:
            try:
                filenames, prompts, neg_prompts = loader.batch(batch_size)
                if not isinstance(prompts, list):
                    prompts = list(prompts)
                if not isinstance(neg_prompts, list):
                    neg_prompts = list(neg_prompts)
                print(f"inference current batch:")
                images = self(prompts, neg_prompts, inference_steps, target_size, base_only, return_type)
                all_images.extend(images)
            except IndexError as e:
                print(e)
                break
        return all_images





def parse_arg():
    parser = argparse.ArgumentParser(prog="Arguments to run SDXL inference pipeline",
                                     description="Arguments to run SDXL inference pipeline")


if __name__ == "__main__":
    sdxl_pipeline = SDXLInferencePipeline()

def parse_arg():
    parser = argparse.ArgumentParser(prog="Arguments to run SDXL inference pipeline",
                                     description="Arguments to run SDXL inference pipeline")
    parser.add_argument("--num_of_inference", type=int, default=50)
    parser.add_argument("--use_refiner", action="store_true")
    parser.add_argument("--output_width", type=int, default=512)
    parser.add_argument("--output_height", type=int, default=512)
    parser.add_argument("--base_model_path", type=str, default="")
    parser.add_argument("--refiner_model_path", type=str, default="")
    parser.add_argument("--cache_model", type=bool, default=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arg()
    sdxl_pipeline = SDXLInferencePipeline()
