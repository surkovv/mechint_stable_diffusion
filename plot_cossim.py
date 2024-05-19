import matplotlib.pyplot as plt
import numpy as np
from autoencoder import *

encoder = AutoEncoder.load(version=21)
log_freqs = np.load('../freqs.npy')
is_rare = (log_freqs < -2.5) & (log_freqs > -6)
rare_enc = encoder.W_enc[:, is_rare]
rare_mean = rare_enc.mean(-1)
is_norm = (log_freqs > -2.5)

plt.rcParams['font.size'] = 16
print(log_freqs.shape)

plt.figure(figsize=(8, 6))
plt.hist(log_freqs, bins=30)
plt.xlabel('Log frequencies')
plt.grid()
plt.savefig('../out.png')

plt.figure(figsize=(8, 6))
cos_sim = rare_mean @ encoder.W_enc / rare_mean.norm() / encoder.W_enc.norm(dim=0)
cos_sim = cos_sim.detach().cpu().numpy()
plt.hist(cos_sim[is_rare], label='Rare', bins=30)
plt.hist(cos_sim[is_norm], label='Norm', bins=30)
plt.grid()
plt.legend()
plt.xlabel('Cosine similarity')
plt.savefig('../cossim.png')