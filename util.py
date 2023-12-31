import json
import os
import random
from argparse import Namespace
from typing import Dict, Optional, Tuple


class PromptLoader:
    def __init__(self, file_path: str | os.PathLike, responsible_slice: list[int, int] | Tuple[int, int] = None):
        self.file_path = file_path
        data: Dict[str: str]
        try:
            with open(file_path, "r") as read_file:
                data = json.load(read_file)
        except OSError as e:
            print(e)
            raise NotImplementedError("currently only support JSON file, or you specify an invalid path")
        except Exception as e:
            print(e)
            raise e
        self.json_dict = data
        self.filenames: Tuple[str] = tuple(data.keys())
        self.prompts: Tuple[str] = tuple(data.values())
        self.processed_head_idx = 0
        if responsible_slice is not None:
            self.responsible_slice = responsible_slice
            self.filenames = self.filenames[responsible_slice[0]: responsible_slice[1]]
            self.prompts = self.prompts[responsible_slice[0]: responsible_slice[1]]
            self.json_dict = dict(zip(self.filenames, self.prompts))

    @classmethod
    def from_namespace(cls, ns: Namespace):
        file_path = ns.prompt_loader_data_file_path
        responsible_slice = ns.responsible_slice
        return cls(file_path, responsible_slice)

    def batch(self, batch_size: Optional[int] = None) -> tuple[tuple, tuple[str, ...], tuple[str, ...]]:
        if self.processed_head_idx >= len(self.json_dict):
            raise IndexError(f"all pairs in prompt loader had been loaded. loaded total number: {len(self.json_dict)}")
        if batch_size is None:
            batch_size = len(self.json_dict) - self.processed_head_idx
        else:
            batch_size = min(batch_size, len(self.json_dict) - self.processed_head_idx)
        print(
            f"cur batch size {batch_size} and selected: {self.processed_head_idx + self.responsible_slice[0]}, {self.processed_head_idx + batch_size + self.responsible_slice[0]}")
        prompt_slice: Tuple[str] = self.prompts[self.processed_head_idx: self.processed_head_idx + batch_size]
        prompts = []
        neg_prompts = []
        for raw_prompt in prompt_slice:
            splitted_raw = raw_prompt.split("\n")
            if len(splitted_raw) == 1:
                prompts.append(splitted_raw[0])
                neg_prompts.append('')
            else:
                prompts.append(splitted_raw[0])
                neg_prompts.append(splitted_raw[1])
        prompts = tuple(prompts)
        neg_prompts = tuple(neg_prompts)

        cur_batch = (self.filenames[self.processed_head_idx: self.processed_head_idx + batch_size],
                     prompts,
                     neg_prompts)
        self.processed_head_idx += batch_size
        return cur_batch

    def reset(self, reset_idx: Optional[int] = None):
        if reset_idx is None or reset_idx < 0:
            reset_idx = 0
        self.processed_head_idx = reset_idx

    def shuffle(self):
        length = len(self.json_dict)
        for idx in range(length):
            cur_flip_idx = random.randint(idx, length - 1)
            self.filenames[idx], self.filenames[cur_flip_idx] = self.filenames[cur_flip_idx], self.filenames[idx]
            self.prompts[idx], self.prompts[cur_flip_idx] = self.prompts[cur_flip_idx], self.prompts[idx]

    def save_images(self, images, path: str, filenames: list[str] = None):
        if not os.path.exists(path):
            # If it doesn't exist, create it
            os.makedirs(path)
        if path[-1] in ("\\", "/"):
            path = path[:-1]
        # assert len(self.filenames) == images, f"try to save images with inconsistent length filename and image lists"
        if filenames is None:
            # by default store from the beginning
            for filename, image in zip(self.filenames, images):
                # assert isinstance(image, PIL.Image.Image), f"get image as class {type(image)} instead of PIL image"
                image.save(f"{path}/{filename}.png", "PNG")
        else:
            for filename, image in zip(filenames, images):
                image.save(f"{path}/{filename}.png", "PNG")

    def get_filenames(self):
        return self.filenames

    def get_json_dict(self):
        return self.json_dict

    def __len__(self):
        return len(self.json_dict)
