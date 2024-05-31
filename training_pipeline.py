from datasets import load_dataset
from daam import trace, set_seed, UNetFFLocator
from diffusers import DiffusionPipeline
from matplotlib import pyplot as plt
import torch
from random import shuffle
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from diffusers.utils import load_image

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
    def __init__(self, restrict=list(range(4, 13)), n_filter=50000):
        self.image = None
        self.prompt = None
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16, use_safetensors=True)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True, safety_checker=None
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.restrict = restrict
        self.pipe = self.pipe.to('cuda')
        self.batch_size = 10
        self.n_filter = n_filter
        self.gen = set_seed(0) 

    def set_prompt(self, prompt) :
        self.prompt = prompt

    def set_image(self, img_path):
        self.image = load_image(img_path)

    def get(self):
        assert self.image is not None, "Image not set"
        result = {}

        with torch.no_grad():
            with trace(self.pipe, result, do_rand=True) as tc:
                out = self.pipe(self.prompt, self.image, num_inference_steps=15, 
                generator=self.gen, num_images_per_prompt=self.batch_size)
        
        return [
            torch.cat(r[:(self.n_filter + self.batch_size - 1) // self.batch_size]).flatten(0, 1)[:self.n_filter] 
            for k, r in result.items()
        ]

    def get_sizes(self):
        locator = UNetFFLocator()
        modules = locator.locate(self.pipe.unet)
        return [module.net[0].proj.in_features for i, module in enumerate(modules)if i in self.restrict]