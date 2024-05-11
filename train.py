from autoencoder import *

encoder = AutoEncoder(cfg)
buffer = Buffer(cfg)
encoder_optim = torch.optim.Adam(encoder.parameters(), lr=1e-4, betas=(0.9, 0.99))

i = 0
try:
    while True:
        acts = buffer.next()

        if acts == None:
            break
        
        loss, x_reconstruct, mid_acts, l2_loss, l1_loss = encoder(acts)
        loss.backward()
        encoder.make_decoder_weights_and_grad_unit_norm()
        encoder_optim.step()
        encoder_optim.zero_grad()
        loss_dict = {"loss": loss.item(), "l2_loss": l2_loss.item(), "l1_loss": l1_loss.item()}
        del loss, x_reconstruct, mid_acts, l2_loss, l1_loss, acts
        if (i) % 100 == 0:
            print(loss_dict)
        if (i+1) % 3000 == 0:
            encoder.save()
        i += 1
finally:
    encoder.save()
# %%