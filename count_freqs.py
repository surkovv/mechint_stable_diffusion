from autoencoder import *
import torch
import matplotlib.pyplot as plt
import numpy as np

encoder = AutoEncoder.load(version=21)
buffer = Buffer(cfg)

# Frequency
@torch.no_grad()
def get_freqs():
    act_freq_scores = torch.zeros(encoder.d_hidden, dtype=torch.float32).cuda()
    total = 0

    for i in range(5000):
        batch = buffer.next()
        hidden = encoder(batch)[2]

        act_freq_scores += (hidden > 0).sum(0)
        total+=hidden.shape[0]
    act_freq_scores /= total
    num_dead = (act_freq_scores==0).float().mean()
    print("Num dead", num_dead)
    return act_freq_scores

freqs = get_freqs()
log_freq = (freqs + 10**-6.5).log10().detach().cpu().numpy()
np.save('../freqs.npy', log_freq)
plt.hist(log_freq, bins=30, density=True)
plt.grid()
plt.savefig('../out.png')