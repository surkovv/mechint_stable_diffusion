from diffusers import DiffusionPipeline
import torch
import numpy as np
from tqdm import tqdm
import argparse
from daam import set_seed, trace_intervention

from autoencoder import *
from training_pipeline import Wrapper
from cfg import cfg, thresholds

parser=argparse.ArgumentParser()
parser.add_argument("prompt", type=str)
parser.add_argument("save_dir", type=str)
parser.add_argument("--ae_version", default=15, type=int)
parser.add_argument("--num_images_per_prompt", default=3, type=int)
args=parser.parse_args()


autoencoders = []
ae_folders = os.listdir(cfg['save_dir'])
wrapper = Wrapper(restrict=list(range(4, 13)))
dims = wrapper.get_sizes()
for i in range(0, len(ae_folders)) :
    ae = AutoEncoder(cfg, dims[i], i)
    ae = ae.load(cfg, args.ae_version)
    autoencoders.append(ae)

pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, safety_checker=None).to('cuda')

def generate_image(pipe, out_name):
    out = pipe(args.prompt, num_inference_steps=15, num_images_per_prompt=args.num_images_per_prompt, generator=set_seed(0))
    for i, img in enumerate(out.images):
        img.save(args.save_dir + out_name + f'_{i}.png')
    
generate_image(pipe, 'no_intervention')

normal_neurons_to_delete = {}
rear_neurons_to_delete = {}

for j in tqdm(range(0, len(autoencoders))) :
    normal_neurons_to_delete = {}
    rear_neurons_to_delete = {}

    freqs = np.load(cfg['save_dir'] + r'ae_' + str(j) + r'/freqs_' + str(args.ae_version) + '.npy')
    normal_neurons_to_delete[j] = []
    rear_neurons_to_delete[j] = []

    for k in range (0, freqs.shape[-1]) :
        if np.log10(freqs[k] + 10e-6) > thresholds[j] :
            normal_neurons_to_delete[j].append(k)
        else : 
            rear_neurons_to_delete[j].append(k)

    def get_ae_mask (neurons_to_zero) :
        reset_masks = []
        for i in range(len(autoencoders)):
            if i not in neurons_to_zero:
                reset_masks.append(None)
            else:
                mask = torch.zeros(autoencoders[i].d_hidden, dtype=torch.bool).to('cuda')
                for neuron in neurons_to_zero[i]:
                    mask[neuron] = 1
                reset_masks.append(mask)
        return reset_masks

    with trace_intervention(pipeline=pipe, autoencoders=autoencoders, reset_masks=get_ae_mask(normal_neurons_to_delete)) as trace:
        generate_image(pipe, 'intervention_zero_normal_ae_' + str(j))

    with trace_intervention(pipeline=pipe, autoencoders=autoencoders, reset_masks=get_ae_mask(rear_neurons_to_delete)) as trace:
        generate_image(pipe, 'intervention_zero_rear_ae_' + str(j))

    all_neurons_to_delete = {}
    all_neurons_to_delete[j] = np.arange(0, freqs.shape[-1]).tolist()
    
    with trace_intervention(pipeline=pipe, autoencoders=autoencoders, reset_masks=get_ae_mask(all_neurons_to_delete)) as trace:
        generate_image(pipe, 'intervention_zero_all_ae_' + str(j))

    no_neurons_to_delete = {}
    no_neurons_to_delete[j] = []

    with trace_intervention(pipeline=pipe, autoencoders=autoencoders, reset_masks=get_ae_mask(no_neurons_to_delete)) as trace:
        generate_image(pipe, 'intervention_zero_no_ae_' + str(j))


    