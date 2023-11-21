import torch
from gigagan_pytorch import (
    GigaGAN,
    ImageDataset
)

import wandb


wandb.init(project="gigagan", job_type="simple", save_code=True)

LOG_EVERY = 10

gan = GigaGAN(
    train_upsampler = True,     # set this to True
    generator = dict(
        style_network = dict(
            dim = 64,
            depth = 4
        ),
        dim = 32,
        image_size = 256,
        input_image_size = 64,
        unconditional = True
    ),
    discriminator = dict(
        dim_capacity = 16,
        dim_max = 512,
        image_size = 256,
        num_skip_layers_excite = 4,
        multiscale_input_resolutions = (128,),
        unconditional = True
    ),
    amp = True,
    log_steps_every=LOG_EVERY
).cuda()

dataset = ImageDataset(
    folder = './ds',
    exts = ['jpg', 'JPEG'],
    image_size = 256
)

dataloader = dataset.get_dataloader(batch_size = 1)

gan.set_dataloader(dataloader)

# training the discriminator and generator alternating
# for 100 steps in this example, batch size 1, gradient accumulated 8 times

gan(
    steps = 100000,
    grad_accum_every = 8
)

