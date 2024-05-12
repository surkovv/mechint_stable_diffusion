from autoencoder import *
import torch
import matplotlib.pyplot as plt
import numpy as np
from training_pipeline import *
import json

encoder = AutoEncoder.load(version=21)
wrapper = Wrapper()

num_iters = 360

acts = []
all_prompts = []

for it in range(num_iters):
    prompts, tensors = wrapper.get_div_prompts()
    for prompt, tensor in zip(prompts, tensors):
        batch_size = 1000
        data = []
        total = 0
        for batch in torch.split(tensor, batch_size, dim=0):
            hidden = encoder(batch)[2].detach().cpu()
            data.append(hidden.sum(dim=0))
            total += hidden.shape[0]
        average_acts = torch.stack(data, dim=0).sum(dim=0) / total
        acts.append(average_acts.numpy())
        all_prompts.append(prompt)

    if (it + 1) % 10 == 0:
        with open("../prompts.json", "w") as f:
            json.dump(all_prompts, f)
        np.save("../activations.npy", np.stack(acts, axis=0))

with open("../prompts.json", "w") as f:
    json.dump(all_prompts, f)

np.save("../activations.npy", np.stack(acts, axis=0))