from diffusers import UNet2DModel
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from datasets import load_dataset
from torchvision import transforms
from functools import partial
import torch
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import DDPMPipeline
import os
from diffusers.models.unets.unet_2d import UNet2DOutput
import random
import math
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from PIL import Image
from dataclasses import dataclass
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from torch_fidelity import calculate_metrics
from torch.optim import AdamW
from diffusers.optimization import get_cosine_schedule_with_warmup
import numpy as np
from diffusers.utils import make_image_grid
import copy


#--------------------------------------- Configurations --------------------------
@dataclass
class TrainingConfig:
    image_size = 32
    batch_size = 64
    num_epochs = 500
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 10
    mixed_precision = "fp16"
    output_dir = "cifar10_15"
    seed = 1211
    eval_batch_size = 64
    gen_num_images = 50000


#---------------------------------------- Dataset ---------------------------------
def do_transform(examples, preprocess):
    images = [preprocess(image.convert('RGB')) for image in examples['img']]
    return {'images': images}

def get_dataloaders(dataset_name='cifar10', batch_size=64, num_workers=2):
    # Load dataset from Hugging Face
    dataset = load_dataset(dataset_name)

    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    no_norm_transform = transforms.Compose([
        transforms.ToTensor()  # No normalization, for FID / IS / KID
    ])

    # Datasets
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    real_dataset = copy.deepcopy(dataset['train'])

    train_dataset.set_transform(partial(do_transform, preprocess=train_transform))
    test_dataset.set_transform(partial(do_transform, preprocess=test_transform))
    real_dataset.set_transform(partial(do_transform, preprocess=no_norm_transform))

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, real_loader


#----------------------------------- Prototype Module ---------------------------------------
class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, dim, 4, 2, 1), nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.encoder(x)  # (B, dim, 1, 1)
        return x.view(x.size(0), -1)  # (B, dim)

    def __call__(self, x):
        return self.forward(x)


class PrototypeModule(nn.Module):
    def __init__(self, k=10, tau=1.0, alpha=1.0, beta=1.0, dim=512, in_channels=3):
        super().__init__()
        self.k = k
        self.tau = tau
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.in_channels=in_channels


        # Prototypes: shape (K, dim)
        self.prototypes = nn.Parameter(torch.randn(self.k, self.dim))
        nn.init.kaiming_uniform_(self.prototypes, a=0.1)

        # Feature extractor f_phi
        self.f_phi = FeatureExtractor(in_channels=self.in_channels, dim=self.dim)

    def select_prototype(self, x_feat):
        """
        Input: x_feat (B, dim)
        Output: closest prototypes (B, dim)
        """

        distances = torch.cdist(x_feat, self.prototypes)  # (B, K)
        indices = distances.argmin(dim=1)  # (B,)
        selected = self.prototypes[indices]  # (B, dim)
        return selected

    def compute_loss(self, x_feat):
        """
        Inputs:
            x: input images (B, dim)
        Returns:
            total_loss (scalar)
        """

        distances = torch.cdist(x_feat, self.prototypes)  # (B, K)
        indices = distances.argmin(dim=1)  # (B,)
        e_selected = self.prototypes[indices]  # (B, dim)

        # Contrastive Loss
        logits = -self.tau * distances  # (B, K)
        labels = indices
        contrastive_loss = F.cross_entropy(logits, labels)

        # Align Loss
        align_loss = F.mse_loss(x_feat, e_selected)

        # Compact Loss
        proto_norm = F.normalize(self.prototypes, dim=1)
        sim_matrix = torch.matmul(proto_norm, proto_norm.T)  # (K, K)
        compact_loss = ((sim_matrix - torch.eye(self.k, device=sim_matrix.device)) ** 2).sum() # exclude diagonal

        return contrastive_loss + self.alpha * align_loss + self.beta * compact_loss

    def select_random_prototype(self, n):
        """
        Return n random prototypes from the bank (n, dim)
        """
        idx = random.choices(range(self.k), k=n)
        return self.prototypes[idx]

    def project_sample(self, x):
        """
        Return feature embedding f_phi(x), avoids repeated computation
        """
        return self.f_phi(x)  # (B, dim)


#------------------------------------------- Unet -------------------------------------------------
class Unet(UNet2DModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bottleneck_attn = nn.MultiheadAttention(
        embed_dim=self.config.block_out_channels[-1],
        num_heads=8,
        batch_first=True
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        prototype: Optional[torch.Tensor] = None,
    ) -> Union[UNet2DOutput, Tuple]:
        r"""
        The [`UNet2DModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch, channel, height, width)`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d.UNet2DOutput`] instead of a plain tuple.

        Returns:
            [`~models.unets.unet_2d.UNet2DOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unets.unet_2d.UNet2DOutput`] is returned, otherwise a `tuple` is
                returned where the first element is the sample tensor.
        """
        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when doing class conditioning")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb
        elif self.class_embedding is None and class_labels is not None:
            raise ValueError("class_embedding needs to be initialized in order to use class conditioning")

        # 2. pre-process
        skip_sample = sample
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # ----------- Cross-Attention after downsampling ------------------
        if prototype is None:
            raise ValueError("You must provide a `prototype` tensor for cross-attention")

        b, c, h, w = sample.shape
        query = sample.view(b, c, h * w).transpose(1, 2)  # (B, HW, C)

        # prototype: expected shape (B, N, C)
        key = value = prototype.to(dtype=sample.dtype)

        attn_output, _ = self.bottleneck_attn(query, key, value)
        attn_output = attn_output.transpose(1, 2).view(b, c, h, w)  # (B, C, H, W)

        # Résiduel
        sample = sample + attn_output
        # ---------------------------------------------------------------


        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(sample, emb)

        # 5. up
        skip_sample = None
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(sample, res_samples, emb, skip_sample)
            else:
                sample = upsample_block(sample, res_samples, emb)

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if skip_sample is not None:
            sample += skip_sample

        if self.config.time_embedding_type == "fourier":
            timesteps = timesteps.reshape((sample.shape[0], *([1] * len(sample.shape[1:]))))
            sample = sample / timesteps

        if not return_dict:
            return (sample,)

        return UNet2DOutput(sample=sample)

    def __call__(self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        prototype: Optional[torch.Tensor] = None,
    ):
        return self.forward(sample, timestep, class_labels, return_dict, prototype)

#------------------------------------- Prototype Diffusion Model -----------------------------------------
class SinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps:int, embed_dim: int):
        super().__init__()

        position = torch.arange(time_steps).unsqueeze(1).float()  # [T, 1]
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))  # [D/2]

        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(position * div)  # pairs
        embeddings[:, 1::2] = torch.cos(position * div)  # impairs

        self.embeddings = embeddings  # [T, D]

    def forward(self, x, t):  # x: [B, C, H, W], t: [B]
        embeds = self.embeddings.to(t.device)[t] # [B, D]
        return embeds

    def __call__(self, x, t):
        return self.forward(x, t)

class PrototypeDiffusionModel(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.prototype = PrototypeModule(
            k=params["k"],
            tau=params["tau"],
            alpha=params["alpha"],
            beta=params["beta"],
            dim=params["dim"],
            in_channels=params["in_channels"]
        )

        self.unet = Unet(
            sample_size=params["sample_size"],
            in_channels=params["in_channels"],
            out_channels=params["out_channels"],
            layers_per_block=params["layers_per_block"],
            block_out_channels=params["block_out_channels"],
            down_block_types=params["down_block_types"],
            up_block_types=params["up_block_types"]
        )

        # New sinusoidal time embedding module
        self.sinusoidal_embedding = SinusoidalEmbeddings(
            time_steps=params.get("time_steps", 1000),         # e.g., 1000 for DDPM
            embed_dim=params.get("time_embed_dim", 1024)       # needs to be ≥ prototype dim
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        image: Optional[torch.Tensor] = None
    ):
        """
        Args:
            sample: Noisy input image (B, C, H, W)
            timestep: Diffusion timestep (int or tensor)
            image: Optional clean image used for prototype selection

        Returns:
            UNet2DOutput
        """
        batch_size = sample.size(0)
        device = sample.device

        # Handle timestep formatting
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], dtype=torch.long, device=device)
        if timestep.ndim == 0:
            timestep = timestep[None]
        if timestep.shape[0] == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size)

        # Compute sinusoidal time embedding
        sin_t_emb = self.sinusoidal_embedding(sample, timestep)  # [B, D]

        # Select or sample prototype
        image_feat = None
        if image is not None:
            image_feat = self.prototype.project_sample(image)             # (B, dim)
            selected_proto = self.prototype.select_prototype(image_feat) # (B, dim)
        else:
            selected_proto = self.prototype.select_random_prototype(batch_size)  # (B, dim)

        selected_proto = selected_proto.to(dtype=sample.dtype, device=device)
        sin_t_emb = sin_t_emb.to(dtype=sample.dtype, device=device)

        # Truncate sinusoidal embedding if needed
        D = selected_proto.shape[1]
        if sin_t_emb.shape[1] >= D:
            sin_t_emb = sin_t_emb[:, :D]
        elif sin_t_emb.shape[1] < D:
            raise ValueError(f"Sinusoidal embedding dim ({sin_t_emb.shape[1]}) is smaller than prototype dim ({D}).")

        # Add temporal embedding to prototype
        proto_with_time = selected_proto + sin_t_emb  # (B, dim)
        proto_with_time = proto_with_time.unsqueeze(1)  # (B, 1, dim)

        if image is None:
            # Forward through U-Net with prototype
            return self.unet(sample=sample, timestep=timestep, prototype=proto_with_time)
        else:
            return (self.unet(sample=sample, timestep=timestep, prototype=proto_with_time),
                    self.prototype.compute_loss(image_feat))


#--------------------------------------------- Output Pipeline --------------------------------
class PDMPipeline:
    def __init__(self, model, scheduler):
        self.model = model  # model = PrototypeDiffusionModel
        self.scheduler = scheduler

        if hasattr(self.scheduler, "set_format"):
            self.scheduler = self.scheduler.set_format("pt")

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        image: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        device: Optional[torch.device] = None,
    ) -> Union[ImagePipelineOutput, Tuple]:

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(device)

        # Initial noise sample
        sample = torch.randn(
            (batch_size,
             self.model.unet.config.in_channels,
             self.model.unet.config.sample_size,
             self.model.unet.config.sample_size),
            generator=generator,
            device=device
        )

        self.scheduler.set_timesteps(1000, device=device)

        for t in self.scheduler.timesteps:
            model_output = self.model(sample=sample, timestep=t, image=image)

            # Si model() retourne un tuple : (output, loss)
            if isinstance(model_output, tuple):
                model_output = model_output[0]  # on prend juste l'image

            sample = self.scheduler.step(model_output.sample, t, sample, generator=generator).prev_sample

        # Normalize to [0,1]
        sample = (sample / 2 + 0.5).clamp(0, 1)

        # Convert to numpy
        sample = sample.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            sample = self.numpy_to_pil(sample)

        if not return_dict:
            return (sample,)

        return ImagePipelineOutput(images=sample)


#------------------------------------ Train the model ----------------------------------------------------

def evaluate(config, epoch, pipeline):
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator = torch.Generator(device=device).manual_seed(config.seed)

        images = pipeline(
            batch_size=config.eval_batch_size,
            image=None,
            output_type="pil",  # images est une liste de PIL.Image
            return_dict=True,
            generator=generator,
        ).images

    to_tensor = transforms.ToTensor()
    images_tensor = torch.stack([to_tensor(img) for img in images])  # [B, C, H, W]

    image_grid = make_grid(images_tensor, nrow=8)

    samples_dir = os.path.join(config.output_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    save_image(image_grid, f'{samples_dir}/epoch_{epoch:04d}_grid.png')
    print(f"Saved evaluation image grid for epoch {epoch}")


def save_model(accelerator, model, output_dir):
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        state_dict = accelerator.get_state_dict(unwrapped_model)
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
        print(f"Model saved to {output_dir}")

def generate_and_save_images(config, pipeline, accelerator, total_images=50000):
    device = accelerator.device
    generated_dir = os.path.join(config.output_dir, "generated")
    os.makedirs(generated_dir, exist_ok=True)


    batch_size = 50
    num_batches = total_images // batch_size

    for i in tqdm(range(num_batches), desc="Generating images"):
        with torch.no_grad():
            generator = torch.Generator(device=device).manual_seed(config.seed + i)

            images = pipeline(
                batch_size=batch_size,
                image=None,
                output_type="pt",
                return_dict=True,
                generator=generator
            ).images

            if isinstance(images, np.ndarray):
                images = torch.from_numpy(images)

            elif isinstance(images, list) and isinstance(images[0], np.ndarray):
                images = torch.stack([torch.from_numpy(img) for img in images])

        images = images.permute(0, 3, 1, 2).cpu()  # B, C, H, W

        for j in range(images.size(0)):
            index = i * batch_size + j
            save_image(images[j], os.path.join(generated_dir, f"image_{index:05d}.png"))

    print(f"Saved {total_images} generated images to {generated_dir}")
    return generated_dir

def save_real_images(config, real_dataloader, total_images = 50000):
    real_dir = os.path.join(config.output_dir, "real")
    os.makedirs(real_dir, exist_ok=True)


    count = 0

    for batch in real_dataloader:
        images = batch['images']  # [B, C, H, W]
        images = images.cpu()
        for i in range(images.size(0)):
            if count >= total_images:
                print(f"Saved exactly {count} real images to {real_dir}")
                return real_dir
            save_image(images[i], os.path.join(real_dir, f"real_{count:05d}.png"))
            count += 1

    print(f"Saved {count} real images to {real_dir}")
    return real_dir


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, real_loader):

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with='tensorboard',
        project_dir=os.path.join(config.output_dir, 'logs')
    )

    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        model.train()
        for step, batch in enumerate(train_dataloader):
            clean_images = batch['images'].to(accelerator.device)
            noise = torch.randn_like(clean_images)
            bs = clean_images.size(0)

            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bs,),
                device=clean_images.device,
                dtype=torch.long,
            )

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):

                noise_pred, loss_proto = model(noisy_images, timesteps, image=clean_images)
                loss_diff = F.mse_loss(noise_pred.sample, noise)
                loss = loss_diff + loss_proto
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            logs = {
                "loss_total": loss.detach().item(),
                "loss_diff": loss_diff.detach().item(),
                "loss_proto": loss_proto.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
                "epoch": epoch,
            }
            progress_bar.update(1)
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
            if accelerator.is_main_process:
                print(f"[Epoch {epoch} | Step {global_step}] -", logs)

        if accelerator.is_main_process:
            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                unwrapped_model = accelerator.unwrap_model(model)
                pipeline = PDMPipeline(model=unwrapped_model, scheduler=noise_scheduler)
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                save_model(accelerator, model, config.output_dir)

    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        pipeline = PDMPipeline(model=unwrapped_model, scheduler=noise_scheduler)
        generated_dir = generate_and_save_images(config, pipeline, accelerator)
        #real_dir = save_real_images(config, real_loader)
        real_dir = "real"

        metrics = calculate_metrics(
            input1=generated_dir,
            input2=real_dir,
            cuda=torch.cuda.is_available(),
            isc=True,
            fid=True,
            kid=True
        )
        print("Evaluation metrics after training:")
        print(metrics)

if __name__ == "__main__":
    config = TrainingConfig()
    params = {
        "sample_size": 32,
        "in_channels": 3,
        "out_channels": 3,
        "layers_per_block": 2,
        "block_out_channels": [128, 256, 256, 256],
        "down_block_types": [
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
            "DownBlock2D"
        ],
        "up_block_types": [
            "UpBlock2D",
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D"
        ],
        "k": 15,
        "tau": 1.0,
        "alpha": 1.0,
        "beta": 1.0,
        "dim": 256,
        "time_steps": 2000,
        "time_embed_dim": 256
    }
    model = PrototypeDiffusionModel(params)
    noise_scheduler = DDPMScheduler(num_train_timesteps=2000)
    train_loader, test_loader, real_loader = get_dataloaders()
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=len(train_loader) * config.num_epochs
    )
    train_loop(config, model, noise_scheduler, optimizer, train_loader, lr_scheduler, real_loader)
