from autoencoder import *
import matplotlib.pyplot as plt
import numpy as np

from training_pipeline import Wrapper
from cfg import cfg

wrapper = Wrapper(restrict=list(range(4, 13)))

class AutoEncoderEnv:
    def __init__(self, dim, num):
        self.dim = dim
        self.autoencoder = AutoEncoder(cfg, dim, num)
        self.encoder_optim = torch.optim.Adam(self.autoencoder.parameters(), lr=3e-5, betas=(0.9, 0.99))
        self.occurs = torch.zeros(self.autoencoder.d_hidden).to(cfg['device'])
        self.total = 0
        self.num = num
    
    def flush_frequencies(self):
        self.occurs = torch.zeros(self.autoencoder.d_hidden).to(cfg['device'])
        self.total = 0
    
    def get_frequencies(self):
        return self.occurs / self.total
    
    def plot_freqs(self):
        freqs = self.get_frequencies()
        log_freq = (freqs + 10**-6.5).log10().detach().cpu().numpy()

        plt.rcParams['font.size'] = 16
        plt.figure(figsize=(8, 6))
        plt.hist(log_freq, bins=30, density=True)
        plt.grid()
        plt.savefig(os.path.join(self.autoencoder.save_dir, f'freqs_plot_{self.autoencoder.get_version() - 1}.png'))

    def plot_cossim(self):
        freqs = self.get_frequencies()
        log_freqs = (freqs + 10**-6.5).log10().detach().cpu().numpy()
        
        is_rare = (log_freqs < -2.5) & (log_freqs > -6)

        if is_rare.sum() == 0:
            return

        rare_enc = self.autoencoder.W_enc[:, is_rare]
        rare_mean = rare_enc.mean(-1)
        is_norm = (log_freqs > -2.5)
        if is_norm.sum() == 0:
            return
        
        plt.rcParams['font.size'] = 16
        plt.figure(figsize=(8, 6))
        cos_sim = rare_mean @ self.autoencoder.W_enc / rare_mean.norm() / self.autoencoder.W_enc.norm(dim=0)
        cos_sim = cos_sim.detach().cpu().numpy()
        plt.hist(cos_sim[is_rare], label='Rare', bins=30)
        plt.hist(cos_sim[is_norm], label='Norm', bins=30)
        plt.grid()
        plt.legend()
        plt.xlabel('Cosine similarity')
        plt.savefig(os.path.join(self.autoencoder.save_dir, f'cossim_plot_{self.autoencoder.get_version() - 1}.png'))


@torch.no_grad()
def re_init(indices, encoder):
    new_W_enc = (torch.nn.init.kaiming_uniform_(torch.zeros_like(encoder.W_enc)))
    new_W_dec = (torch.nn.init.kaiming_uniform_(torch.zeros_like(encoder.W_dec)))
    new_b_enc = (torch.zeros_like(encoder.b_enc))
    print(new_W_dec.shape, new_W_enc.shape, new_b_enc.shape)
    encoder.W_enc.data[:, indices] = new_W_enc[:, indices]
    encoder.W_dec.data[indices, :] = new_W_dec[indices, :]
    encoder.b_enc.data[indices] = new_b_enc[indices]

dims = wrapper.get_sizes()
print(dims)
ae_envs = [AutoEncoderEnv(dim, i) for i, dim in enumerate(dims)]
buffers_handler = BuffersHandler(cfg, wrapper)

def need_count_freqs(i):
    return i % cfg['iters_to_save'] >= cfg['iters_to_save'] // 2

i = 0

losses, l1_losses, l2_losses = [], [], []

try:
    while True:
        acts_many = buffers_handler.next()

        if acts_many == None:
            break
        print(i)
        for ne, (ae_env, acts) in enumerate(zip(ae_envs, acts_many)):
            encoder = ae_env.autoencoder
            encoder_optim = ae_env.encoder_optim

            loss, x_reconstruct, mid_acts, l2_loss, l1_loss = encoder(acts)
            loss.backward()
            encoder.make_decoder_weights_and_grad_unit_norm()
            encoder_optim.step()
            encoder_optim.zero_grad()

            if need_count_freqs(i):
                ae_env.total += mid_acts.shape[0]
                ae_env.occurs += (mid_acts > 0).sum(0)

            losses.append(loss.item())
            l2_losses.append(l2_loss.item())
            l1_losses.append(l1_loss.item())
            del loss, x_reconstruct, mid_acts, l2_loss, l1_loss, acts
            if (i) % cfg['iters_to_log'] == 0:
                print(f'Iter {i} Encoder {ne}; dim {ae_env.dim}')
                loss_dict = {"loss": np.mean(losses), "l2_loss": np.mean(l2_losses), "l1_loss": np.mean(l1_losses)}
                print(loss_dict)
                losses, l1_losses, l2_losses = [], [], []
            if (i+1) % cfg['iters_to_save'] == 0:
                encoder.save()
                freqs = ae_env.get_frequencies()
                rare_mask = freqs < cfg['rare_threshold']
                np.save(os.path.join(encoder.save_dir, f'freqs_{encoder.get_version() - 1}.npy'), freqs.cpu().numpy())
                print(f"Encoder {ne}. Rare {rare_mask.to(dtype=float).mean():.3f}")
                ae_env.plot_freqs()
                ae_env.plot_cossim()
                if cfg['do_resampling']:
                    re_init(rare_mask, encoder)
        i += 1
finally:
    pass
    for ae_env in ae_envs:
        ae_env.autoencoder.save()
# %%