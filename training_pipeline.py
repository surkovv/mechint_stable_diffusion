from datasets import load_dataset
from daam import trace, set_seed
from diffusers import DiffusionPipeline
from matplotlib import pyplot as plt
import torch


class TrainingPipeline:
    def __init__(self, dataset, pipe, n_filter=10000):
        self.dataset = dataset
        self.pipe = pipe
        self.current_idx = 0
        self.batch_size = 10
        self.gen = set_seed(0) 
        self.n_filter = n_filter
        self.pipe.safety_checker = lambda images, clip_input: (images, [False] * len(images))

    def get(self):
        result = {}
        prompts = self.dataset[self.current_idx: self.current_idx + self.batch_size]
        self.current_idx += self.batch_size

        if self.current_idx >= len(self.dataset):
            return None

        with torch.no_grad():
            with trace(self.pipe, result) as tc:
                out = self.pipe(prompts, num_inference_steps=30, generator=self.gen)
        return result[7][:self.n_filter]


class Wrapper:
    def __init__(self):
        coco = load_dataset("embedding-data/coco_captions_quintets")
        dataset = [st['set'][0] for st in coco['train']]
        model_id = 'CompVis/stable-diffusion-v1-4'
        device = 'cuda'

        pipe = DiffusionPipeline.from_pretrained(model_id, use_auth_token=True, torch_dtype=torch.float16, use_safetensors=True, variant='fp16')
        pipe = pipe.to(device)
        self.tp = TrainingPipeline(dataset, pipe)

    def get(self):
        return self.tp.get()


if __name__ == '__main__':
    wrapper = Wrapper()
    A = wrapper.get()
    print(A.shape)