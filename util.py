import os
import json
import random
from typing import Dict, Optional


class PromptLoader:
    def __init__(self, file_path: str | os.PathLike):
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
        self.filenames = tuple(data.keys())
        self.prompts = tuple(data.values())
        self.processed_head_idx = 0

    def batch(self, batch_size: Optional[int] = None):
        if self.processed_head_idx >= len(self.json_dict):
            raise IndexError(f"all pairs in prompt loader had been loaded.")
        if batch_size is None:
            batch_size = len(self.json_dict) - self.processed_head_idx
        else:
            assert batch_size <= len(self.json_dict) - self.processed_head_idx, \
                (f"specified batch size {batch_size} greater than prompt loader remaining "
                 f"cached prompts: {len(self.json_dict - self.processed_head_idx)}.")
        cur_batch = (self.filenames[self.processed_head_idx: self.processed_head_idx+batch_size],
                     self.prompts[self.processed_head_idx: self.processed_head_idx+batch_size])
        self.processed_head_idx += batch_size
        return cur_batch

    def reset(self, reset_idx: Optional[int] = None):
        if reset_idx is None or reset_idx < 0:
            reset_idx = 0
        self.processed_head_idx = reset_idx

    def shuffle(self):
        length = len(self.json_dict)
        for idx in range(length):
            cur_flip_idx = random.randint(idx, length-1)
            self.filenames[idx], self.filenames[cur_flip_idx] = self.filenames[cur_flip_idx], self.filenames[idx]
            self.prompts[idx], self.prompts[cur_flip_idx] = self.prompts[cur_flip_idx], self.prompts[idx]



