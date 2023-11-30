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


torch.manual_seed(0)

# wandb.init(project="gigagan", job_type="flowers100", save_code=True)

LOG_EVERY = 10

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
    save_and_sample_every = 10000
).cuda()

checkpoint = torch.load("gigagan-models/model-7.ckpt")

gan.G.load_state_dict(checkpoint['G'])
gan.G_ema.load_state_dict(checkpoint['G_ema'])
# gan.D.load_state_dict(checkpoint['D'])

dataset = ImageDataset(
    folder = './ds100',
    exts = ['jpg', 'JPEG'],
    image_size = 256
)

dataloader = dataset.get_dataloader(batch_size = 1)

gan.set_dataloader(dataloader)

testset = ImageDataset(
    folder = './dstest',
    exts = ['jpg', 'JPEG'],
    image_size = 64
)
# lowres = testset[0].cuda()
dl = testset.get_dataloader(batch_size=8)
lowres = next(iter(dl)).cuda()

images = gan.generate(lowres) # (1, 3, 256, 256)

# us = torch.nn.Upsample(scale_factor = 4, mode = 'bilinear', align_corners = False)
# lowres_upsample = us(lowres)

res_shape = (256, 256)
# pil_img = ToPILImage(lowres)
# pil_image_scaled = pil_img.resize(res_shape, Image.BILINEAR)
torch_img_scaled = F.interpolate(lowres, res_shape, mode='bilinear')

compare = torch.cat((torch_img_scaled, images))

save_image(lowres, "gigagan-results/lowres.jpg")
save_image(images, "gigagan-results/generated.jpg")
# save_image(lowres_upsample, "gigagan-results/lowres_upsample.jpg")
# save_image(pil_image_scaled, "gigagan-results/pil_image_scaled.jpg")
save_image(torch_img_scaled, "gigagan-results/torch_img_scaled.jpg")
save_image(compare, "gigagan-results/compare.jpg")
