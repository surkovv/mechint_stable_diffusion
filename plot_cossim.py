import matplotlib.pyplot as plt
import numpy as np
from autoencoder import *

encoder = AutoEncoder.load(version=21)
log_freqs = np.load('../freqs.npy')
is_rare = (log_freqs < -2.5) & (log_freqs > -6)
rare_enc = encoder.W_enc[:, is_rare]
rare_mean = rare_enc.mean(-1)
is_norm = (log_freqs > -2.5)

cos_sim = rare_mean @ encoder.W_enc / rare_mean.norm() / encoder.W_enc.norm(dim=0)
cos_sim = cos_sim.detach().cpu().numpy()
plt.hist(cos_sim[is_rare], label='Rare', bins=30)
plt.hist(cos_sim[is_norm], label='Norm', bins=30)
plt.grid()
plt.legend()
plt.savefig('../cossim.png')