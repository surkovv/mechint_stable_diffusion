from autoencoder import *

class AutoEncoderEnv:
    def __init__(self, dim, num):
        self.dim = dim
        self.autoencoder = AutoEncoder(cfg, dim, num)
        self.encoder_optim = torch.optim.Adam(encoder.parameters(), lr=1e-4, betas=(0.9, 0.99))
        self.occurs = torch.zeros(self.autoencoder.d_hidden)
        self.total = 0
    
    def flush_frequencies(self):
        self.occurs = torch.zeros(self.autoencoder.d_hidden)
        self.total = 0
    
    def get_frequencies(self):
        return self.occurs / self.total

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
ae_envs = [AutoEncoderEnv(dim, i) for i, dim in enumerate(dims)]
buffers_handler = BuffersHandler(cfg)

def need_count_freqs(i):
    return i % 3000 >= 1500

i = 0
try:
    while True:
        acts_many = buffers_handler.next()

        if acts_many == None:
            break

        for ne, (ae_env, acts) in enumerate(zip(ae_envs, acts_many)):
            encoder = ae_env.encoder
            encoder_optim = ae_env.encoder_optim

            loss, x_reconstruct, mid_acts, l2_loss, l1_loss = encoder(acts)
            loss.backward()
            encoder.make_decoder_weights_and_grad_unit_norm()
            encoder_optim.step()
            encoder_optim.zero_grad()

            if need_count_freqs(i):
                ae_env.total += mid_acts.shape[0]
                ae_env.occurs += (mid_acts > 0).sum(0)

            loss_dict = {"loss": loss.item(), "l2_loss": l2_loss.item(), "l1_loss": l1_loss.item()}
            del loss, x_reconstruct, mid_acts, l2_loss, l1_loss, acts
            if (i) % 100 == 0:
                print(f'Iter {i} Encoder {ne}; dim {dim}')
                print(loss_dict)
            if (i) % 3000 == 0:
                encoder.save()
                if cfg['do_resampling']:
                    freqs = ae_env.get_frequencies()
                    rare_mask = freqs < cfg['rare_threshold']
                    re_init(rare_mask, encoder)
        i += 1
finally:
    encoder.save()
# %%