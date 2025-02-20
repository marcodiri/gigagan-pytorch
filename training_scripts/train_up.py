import torch
from torchvision.utils import save_image
from gigagan_pytorch import (
    GigaGAN,
    ImageDataset
)

import wandb


wandb.init(project="gigagan", job_type="flowers7000", save_code=True)

LOG_EVERY = 100

gan = GigaGAN(
    train_upsampler = True,     # set this to True
    generator = dict(
        style_network = dict(
            # style network è un MLP con `depth` livelli
            # e dim è la dimensione di input e output
            # quindi 64 se l'input_image_size è 64px
            dim = 64,
            depth = 4
        ),
        dim = 32,
        image_size = 256,
        input_image_size = 64,
        # dim_mults = (1, 2, 4, 8, 16),
        # full_attn = (False, False, False, True, True),
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
    log_steps_every=LOG_EVERY,
    save_and_sample_every = 20000
).cuda()

dataset = ImageDataset(
    folder = './ds',
    exts = ['jpg', 'JPEG'],
    image_size = 256
)

dataloader = dataset.get_dataloader(batch_size = 1)

gan.set_dataloader(dataloader)

gan(
    steps = 200000,
    grad_accum_every = 1
)
