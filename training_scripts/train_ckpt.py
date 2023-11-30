import torch
from PIL import Image
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.transforms import ToPILImage
from gigagan_pytorch import (
    GigaGAN,
    ImageDataset
)

import wandb


torch.manual_seed(1)

wandb.init(project="gigagan", job_type="flowers100", save_code=True)

LOG_EVERY = 50

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
    learning_rate = 2e-5,
    amp = True,
    log_steps_every=LOG_EVERY,
    save_and_sample_every = 10000
).cuda()

checkpoint = torch.load("gigagan-models/model-7_ds100_70k.ckpt")

gan.G.load_state_dict(checkpoint['G'])
gan.G_ema.load_state_dict(checkpoint['G_ema'])
gan.D.load_state_dict(checkpoint['D'])
gan.G_opt.load_state_dict(checkpoint['G_opt'])
gan.D_opt.load_state_dict(checkpoint['D_opt'])
gan.G_opt.scaler.load_state_dict(checkpoint['G_scaler'])
gan.D_opt.scaler.load_state_dict(checkpoint['D_scaler'])
gan.steps[0] = checkpoint['steps']

dataset = ImageDataset(
    folder = './ds100',
    exts = ['jpg', 'JPEG'],
    image_size = 256
)

dataloader = dataset.get_dataloader(batch_size = 1)

gan.set_dataloader(dataloader)

gan(
    steps = 200000,
    grad_accum_every = 1
)
