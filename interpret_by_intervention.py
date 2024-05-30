from datasets import load_dataset
from daam import trace, set_seed, UNetFFLocator, trace_intervention
from diffusers import DiffusionPipeline
from matplotlib import pyplot as plt
import torch
from random import shuffle
from diffusers import StableDiffusionImg2ImgPipeline, ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from diffusers.utils import make_image_grid, load_image
import numpy as np
from tqdm import tqdm
from autoencoder import *

ae_version = 15
ae_folders = os.listdir(cfg['save_dir'])
dims = wrapper.get_sizes()
autoencoders = []

for i in range(0, len(ae_folders)) :
    ae = AutoEncoder(cfg, dims[i], i)
    ae = ae.load(ae_version)
    autoencoders.append(ae)

pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, safety_checker=None).to('cuda')
prompt = "a green table in a kitchen"

num_images_per_prompt = 3

def generate_image(pipe, out_name):
    out = pipe(prompt, num_inference_steps=15, num_images_per_prompt=num_images_per_prompt, generator=set_seed(0))
    for i, img in enumerate(out.images):
        img.save('green_imgs/' + out_name + f'_{i}.png')
    
generate_image(pipe, 'no_intervention')


# thresholds = [-3, -3, -2, -2.5, -2.5, -3.5, -3, -4, -4]
thresholds = [-2, -3, -2, -2, -3, -4, -3, -3, -3]

# normal_neurons_to_delete = {}
# rear_neurons_to_delete = {}
# for j in range (0, 9) :
#     freqs_green_name = 'freqs_green.npy'
#     freqs_clear_name = 'freqs_table.npy'
#     freqs_green = np.load(r'./checkpoints/ae_' + str(j) + r'/ctrl_freqs/' + freqs_green_name)
#     freqs_clear = np.load(r'./checkpoints/ae_' + str(j) + r'/ctrl_freqs/' + freqs_clear_name)

#     all_neurons = np.arange(0, freqs_green.shape[-1])
#     normal_neurons_to_delete[j] = []
#     rear_neurons_to_delete[j] = []
#     mult_threshold = 10

#     for k in range (0, freqs_green.shape[-1]) :
#         if freqs_green[k] > freqs_clear[k] * mult_threshold  :
#             if np.log10(freqs_green[k] + 10e-6) > thresholds[j] :
#                 normal_neurons_to_delete[j].append(k)
#             else : 
#                 rear_neurons_to_delete[j].append(k)

    # normal_neurons_to_delete[j] = all_neurons[np.log10(freqs + 10 ** -6) > thresholds[j]].tolist()
    # rear_neurons_to_delete[j] = all_neurons[np.log10(freqs + 10 ** -6) < thresholds[j]].tolist()

for j in range (0, 9) :
    normal_neurons_to_delete = {}
    rear_neurons_to_delete = {}
    # freqs_name = 'freqs_6.npy'
    # freqs = np.load(r'./chk_residual/ae_' + str(j) + r'/' + freqs_name)

    # all_neurons = np.arange(0, freqs.shape[-1])

    # normal_neurons_to_delete[j] = all_neurons[np.log10(freqs + 10 ** -6) > thresholds[j]].tolist()
    # rear_neurons_to_delete[j] = all_neurons[np.log10(freqs + 10 ** -6) < thresholds[j]].tolist()

    freqs_green_name = 'freqs_green.npy'
    freqs_clear_name = 'freqs_table.npy'
    freqs_green = np.load(r'./checkpoints/ae_' + str(j) + r'/ctrl_freqs/' + freqs_green_name)
    freqs_clear = np.load(r'./checkpoints/ae_' + str(j) + r'/ctrl_freqs/' + freqs_clear_name)

    all_neurons = np.arange(0, freqs_green.shape[-1])
    normal_neurons_to_delete[j] = []
    rear_neurons_to_delete[j] = []
    mult_threshold = 100

    for k in range (0, freqs_green.shape[-1]) :
        if freqs_green[k] > freqs_clear[k] * mult_threshold  :
            if np.log10(freqs_green[k] + 10e-6) > thresholds[j] :
                normal_neurons_to_delete[j].append(k)
            else : 
                rear_neurons_to_delete[j].append(k)

    print("normal_neurons_to_delete ", j, ":", normal_neurons_to_delete)

    my_neurons = normal_neurons_to_delete
    reset_masks = []
    for i in range(len(autoencoders)):
        if i not in my_neurons:
            reset_masks.append(None)
        elif not my_neurons[i] :
            reset_masks.append(None)
        else:
            mask = torch.zeros(autoencoders[i].d_hidden, dtype=torch.bool).to('cuda')
            for neuron in my_neurons[i]:
                mask[neuron] = 1
            reset_masks.append(mask)

    with trace_intervention(pipeline=pipe, autoencoders=autoencoders, reset_masks=reset_masks) as trace:
        generate_image(pipe, 'intervention_zeronormal_ae_' + str(j))

    my_neurons = rear_neurons_to_delete
    reset_masks = []
    for i in range(len(autoencoders)):
        if i not in my_neurons:
            reset_masks.append(None)
        elif not my_neurons[i] :
            reset_masks.append(None)
        else:
            mask = torch.zeros(autoencoders[i].d_hidden, dtype=torch.bool).to('cuda')
            for neuron in my_neurons[i]:
                mask[neuron] = 1
            reset_masks.append(mask)

    with trace_intervention(pipeline=pipe, autoencoders=autoencoders, reset_masks=reset_masks) as trace:
        generate_image(pipe, 'intervention_zerorear_ae_' + str(j))
    

