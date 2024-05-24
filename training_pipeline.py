from datasets import load_dataset
from daam import trace, set_seed, UNetFFLocator
from diffusers import DiffusionPipeline
from matplotlib import pyplot as plt
import torch
from random import shuffle
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image

class TrainingPipeline:
    def __init__(self, dataset, pipe, n_filter=50000):
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
            with trace(self.pipe, result, do_rand=True) as tc:
                out = self.pipe(prompts, num_inference_steps=15, generator=self.gen)
        
        return [
            torch.cat(r[:(self.n_filter + self.batch_size - 1) // self.batch_size]).flatten(0, 1)[:self.n_filter] 
            for k, r in result.items()
        ]

    def get_div_prompts(self):
        prompts = self.dataset[self.current_idx: self.current_idx + self.batch_size]
        self.current_idx += self.batch_size

        if self.current_idx >= len(self.dataset):
            return None

        result = {}

        with torch.no_grad():
            with trace(self.pipe, result, do_rand=False) as tc:
                out = self.pipe(prompts, num_inference_steps=15, generator=self.gen)

        to_return = list(torch.chunk(result[7], len(prompts), dim=0))
        for i, chunk in enumerate(to_return):
            idx =  torch.randperm(chunk.shape[0])
            to_return[i] = chunk[idx][:5000]

        return prompts, to_return

    def get_sizes(self, restrict):
        locator = UNetFFLocator()
        modules = locator.locate(self.pipe.unet)
        return [module.net[0].proj.in_features for i, module in enumerate(modules) if i in restrict]


class Wrapper:
    def __init__(self, restrict):
        coco = load_dataset("embedding-data/coco_captions_quintets")
        dataset = [st['set'][0] for st in coco['train']]
        shuffle(dataset)
        model_id = 'CompVis/stable-diffusion-v1-4'
        device = 'cuda'
        self.restrict = restrict
        pipe = DiffusionPipeline.from_pretrained(model_id, use_auth_token=True, torch_dtype=torch.float16, use_safetensors=True, variant='fp16')
        pipe = pipe.to(device)
        self.tp = TrainingPipeline(dataset, pipe)

    def get(self):
        return self.tp.get()

    def get_div_prompts(self):
        return self.tp.get_div_prompts()

    def get_sizes(self):
        return self.tp.get_sizes(self.restrict)


class AnalysisPipeline:
    def __init__(self, prompt, n_filter=50000):
        self.image = None
        self.prompt = prompt
        self.pipe = AutoPipelineForImage2Image.from_pretrained('CompVis/stable-diffusion-v1-4', safety_checker=None, use_auth_token=True, variant='fp16')
        self.pipe = self.pipe.to('cuda')
        self.pipe.enable_xformers_memory_efficient_attention()
        self.batch_size = 10
        self.n_filter = n_filter
        self.gen = set_seed(0) 


    def set_image(self, img_path):
        self.image = load_image(img_path).to('cuda')

    def get(self):
        assert self.image is not None, "Image not set"
        result = {}

        with torch.no_grad():
            with trace(self.pipe, result, do_rand=True) as tc:
                out = self.pipe(prompt, image, num_inference_steps=15, 
                generator=self.gen, strength=0.5, num_images_per_prompt=self.batch_size)
        
        return [
            torch.cat(r[:(self.n_filter + self.batch_size - 1) // self.batch_size]).flatten(0, 1)[:self.n_filter] 
            for k, r in result.items()
        ]
        

if __name__ == '__main__':
    wrapper = Wrapper()
    A = wrapper.get()
    print(A.shape)