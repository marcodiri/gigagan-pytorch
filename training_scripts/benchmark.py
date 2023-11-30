import torch
from torchvision.utils import save_image
from torchsummary import summary
from gigagan_pytorch import (
    GigaGAN,
    ImageDataset
)
from numerize import numerize

import wandb


LOG_EVERY = 100

dim_mults_list = ((1, 2, 4), (1, 2, 4, 8), (1, 2, 4, 8, 16))

for dim_mults in dim_mults_list:
    wandb.init(project="gigagan_benchmark", job_type="benchmark", save_code=True)

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
            # dim_mults = dim_mults,
            # full_attn = (False,)*len(dim_mults),
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
        save_and_sample_every = 10000,
        model_folder = './gigagan-models-bench',
        results_folder = './gigagan-results-bench',
    ).cuda()

    wandb.log({
        'generator_params': int(gan.G.total_params),
        'discriminator_params': int(gan.D.total_params)
        })
    with open("gigagan-results/model.txt", "w") as f:
        print(summary(gan, steps = 100), file=f)
        print("\n", file=f)
        print(gan, file=f)
    wandb.save("gigagan-results/model.txt")

    dataset = ImageDataset(
        folder = './ds100',
        exts = ['jpg', 'JPEG'],
        image_size = 256
    )

    dataloader = dataset.get_dataloader(batch_size = 1)

    gan.set_dataloader(dataloader)

    gan(
        steps = 100,
        grad_accum_every = 1
    )
