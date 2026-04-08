"""
This file contains an implementation of the pseudocode from the paper
"Simple Diffusion: End-to-End Diffusion for High Resolution Images"
by Emiel Hoogeboom, Tim Salimans, and Jonathan Ho.

Reference:
Hoogeboom, E., Salimans, T., & Ho, J. (2023).
Simple Diffusion: End-to-End Diffusion for High Resolution Images.
Retrieved from https://arxiv.org/abs/2301.11093
"""

import torch
import torch.nn as nn
from torch.special import expm1
import math


# helper
def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


class simpleDiffusion(nn.Module):
    def __init__(
        self,
        unet,
        image_size,
        noise_size=64,
        pred_param="v",
        schedule="shifted_cosine",
        steps=512,
    ):
        super().__init__()

        # Training objective
        assert pred_param in [
            "v",
            "eps",
        ], "Invalid prediction parameterization. Must be 'v' or 'eps'"
        self.pred_param = pred_param

        # Sampling schedule
        assert schedule in [
            "cosine",
            "shifted_cosine",
        ], "Invalid schedule. Must be 'cosine' or 'shifted_cosine'"
        self.schedule = schedule
        self.noise_d = noise_size
        self.image_d = image_size

        # Model
        assert isinstance(
            unet, nn.Module
        ), "Model must be an instance of torch.nn.Module."
        self.model = unet

        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Number of parameters: {num_params}")

        # Steps
        self.steps = steps

    def diffuse(self, x, alpha_t, sigma_t):
        """
        Function to diffuse the input tensor x to a timepoint t with the given alpha_t and sigma_t.

        Args:
        x (torch.Tensor): The input tensor to diffuse.
        alpha_t (torch.Tensor): The alpha value at timepoint t.
        sigma_t (torch.Tensor): The sigma value at timepoint t.

        Returns:
        z_t (torch.Tensor): The diffused tensor at timepoint t.
        eps_t (torch.Tensor): The noise tensor at timepoint t.
        """
        eps_t = torch.randn_like(x)

        z_t = alpha_t * x + sigma_t * eps_t

        return z_t, eps_t

    def logsnr_schedule_cosine(self, t, logsnr_min=-15, logsnr_max=15):
        """
        Function to compute the logSNR schedule at timepoint t with cosine:

        logSNR(t) = -2 * log (tan (pi * t / 2))

        Taking into account boundary effects, the logSNR value at timepoint t is computed as:

        logsnr_t = -2 * log(tan(t_min + t * (t_max - t_min)))

        Args:
        t (int): The timepoint t.
        logsnr_min (int): The minimum logSNR value.
        logsnr_max (int): The maximum logSNR value.

        Returns:
        logsnr_t (float): The logSNR value at timepoint t.
        """
        logsnr_max = logsnr_max + math.log(self.noise_d / self.image_d)
        logsnr_min = logsnr_min + math.log(self.noise_d / self.image_d)
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))

        logsnr_t = -2 * log(torch.tan(torch.tensor(t_min + t * (t_max - t_min))))

        return logsnr_t

    def logsnr_schedule_cosine_shifted(self, t):
        """
        Function to compute the logSNR schedule at timepoint t with shifted cosine:

        logSNR_shifted(t) = logSNR(t) + 2 * log(noise_d / image_d)

        Args:
        t (int): The timepoint t.
        image_d (int): The image dimension.
        noise_d (int): The noise dimension.

        Returns:
        logsnr_t_shifted (float): The logSNR value at timepoint t.
        """
        logsnr_t = self.logsnr_schedule_cosine(t)
        logsnr_t_shifted = logsnr_t + 2 * math.log(self.noise_d / self.image_d)

        return logsnr_t_shifted

    def clip(self, x):
        """
        Function to clip the input tensor x to the range [-1, 1].

        Args:
        x (torch.Tensor): The input tensor to clip.

        Returns:
        x (torch.Tensor): The clipped tensor.
        """
        return torch.clamp(x, -1, 1)

    @torch.no_grad()
    def ddpm_sampler_step(self, z_t, pred, logsnr_t, logsnr_s):
        """
        Function to perform a single step of the DDPM sampler.

        Args:
        z_t (torch.Tensor): The diffused tensor at timepoint t.
        pred (torch.Tensor): The predicted value from the model (v or eps).
        logsnr_t (float): The logSNR value at timepoint t.
        logsnr_s (float): The logSNR value at the sampling timepoint s.

        Returns:
        z_s (torch.Tensor): The diffused tensor at sampling timepoint s.
        """
        c = -expm1(logsnr_t - logsnr_s)
        alpha_t = torch.sqrt(torch.sigmoid(logsnr_t))
        alpha_s = torch.sqrt(torch.sigmoid(logsnr_s))
        sigma_t = torch.sqrt(torch.sigmoid(-logsnr_t))
        sigma_s = torch.sqrt(torch.sigmoid(-logsnr_s))

        if self.pred_param == "v":
            x_pred = alpha_t * z_t - sigma_t * pred
        elif self.pred_param == "eps":
            x_pred = (z_t - sigma_t * pred) / alpha_t

        x_pred = self.clip(x_pred)

        mu = alpha_s * (z_t * (1 - c) / alpha_t + c * x_pred)
        variance = (sigma_s**2) * c

        return mu, variance

    @torch.no_grad()
    def sample(self, x):
        """
        Standard DDPM sampling procedure. Begun by sampling z_T ~ N(0, 1)
        and then repeatedly sampling z_s ~ p(z_s | z_t)

        Args:
        x_shape (tuple): The shape of the input tensor.

        Returns:
        x_pred (torch.Tensor): The predicted tensor.
        """
        z_t = torch.randn(x.shape).to(x.device)

        # Steps T -> 1
        for t in reversed(range(1, self.steps + 1)):
            u_t = t / self.steps
            u_s = (t - 1) / self.steps

            if self.schedule == "cosine":
                logsnr_t = self.logsnr_schedule_cosine(u_t)
                logsnr_s = self.logsnr_schedule_cosine(u_s)
            elif self.schedule == "shifted_cosine":
                logsnr_t = self.logsnr_schedule_cosine_shifted(u_t)
                logsnr_s = self.logsnr_schedule_cosine_shifted(u_s)

            logsnr_t = logsnr_t.to(x.device)
            logsnr_s = logsnr_s.to(x.device)

            pred = self.model(z_t, logsnr_t)
            mu, variance = self.ddpm_sampler_step(
                z_t, pred, torch.tensor(logsnr_t), torch.tensor(logsnr_s)
            )
            z_t = mu + torch.randn_like(mu) * torch.sqrt(variance)

        # Final step
        if self.schedule == "cosine":
            logsnr_1 = self.logsnr_schedule_cosine(1 / self.steps)
            logsnr_0 = self.logsnr_schedule_cosine(0)
        elif self.schedule == "shifted_cosine":
            logsnr_1 = self.logsnr_schedule_cosine_shifted(1 / self.steps)
            logsnr_0 = self.logsnr_schedule_cosine_shifted(0)

        logsnr_1 = logsnr_1.to(x.device)
        logsnr_0 = logsnr_0.to(x.device)

        pred = self.model(z_t, logsnr_1)
        x_pred, _ = self.ddpm_sampler_step(
            z_t, pred, torch.tensor(logsnr_1), torch.tensor(logsnr_0)
        )

        x_pred = self.clip(x_pred)

        # Convert x_pred to the range [0, 1]
        x_pred = (x_pred + 1) / 2

        return x_pred

    def loss(self, x):
        """
        A function to compute the loss of the model. The loss is computed as the mean squared error
        between the predicted noise tensor and the true noise tensor. Various prediction parameterizations
        imply various weighting schemes as outlined in Kingma et al. (2023)

        Args:
        x (torch.Tensor): The input tensor.

        Returns:
        loss (torch.Tensor): The loss value.
        """
        t = torch.rand(x.shape[0])

        if self.schedule == "cosine":
            logsnr_t = self.logsnr_schedule_cosine(t)
        elif self.schedule == "shifted_cosine":
            logsnr_t = self.logsnr_schedule_cosine_shifted(t)

        logsnr_t = logsnr_t.to(x.device)
        alpha_t = torch.sqrt(torch.sigmoid(logsnr_t)).view(-1, 1, 1, 1).to(x.device)
        sigma_t = torch.sqrt(torch.sigmoid(-logsnr_t)).view(-1, 1, 1, 1).to(x.device)
        z_t, eps_t = self.diffuse(x, alpha_t, sigma_t)
        pred = self.model(z_t, logsnr_t)

        if self.pred_param == "v":
            eps_pred = sigma_t * z_t + alpha_t * pred
        else:
            eps_pred = pred

        # Apply min-SNR weighting (https://arxiv.org/pdf/2303.09556)
        snr = torch.exp(logsnr_t).clamp_(max=5)
        if self.pred_param == "v":
            weight = 1 / (1 + snr)
        else:
            weight = 1 / snr

        weight = weight.view(-1, 1, 1, 1)

        loss = torch.mean(weight * (eps_pred - eps_t) ** 2)

        return loss

    # def train_loop(self, config, optimizer, train_dataloader, lr_scheduler):
    #     """
    #     A function to train the model.

    #     Args:
    #     optimizer (torch.optim.Optimizer): The optimizer to use for training.
    #     """
    #     # Initialize accelerator
    #     accelerator = Accelerator(
    #         mixed_precision=config.mixed_precision,
    #         gradient_accumulation_steps=config.gradient_accumulation_steps,
    #         project_dir=os.path.join(config.output_dir, "logs"),
    #     )
    #     if accelerator.is_main_process:
    #         if config.output_dir is not None:
    #             os.makedirs(config.output_dir, exist_ok=True)
    #         if config.push_to_hub:
    #             repo_id = create_repo(
    #                 repo_id=config.hub_model_id or Path(config.output_dir).name,
    #                 exist_ok=True,
    #             ).repo_id
    #             print(repo_id)

    #     model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    #         self.model, optimizer, train_dataloader, lr_scheduler
    #     )

    #     # Create an EMA model
    #     ema = EMA(model, beta=0.9999, update_after_step=100, update_every=10)

    #     global_step = 0

    #     for epoch in range(config.num_epochs):
    #         progress_bar = tqdm(total=len(train_dataloader))
    #         progress_bar.set_description(f"Epoch {epoch}")

    #         for step, batch in enumerate(train_dataloader):
    #             x = batch["images"]

    #             with accelerator.accumulate(model):
    #                 loss = self.loss(x)
    #                 loss = loss.to(next(model.parameters()).dtype)
    #                 accelerator.backward(loss)
    #                 accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
    #                 optimizer.step()
    #                 lr_scheduler.step()
    #                 optimizer.zero_grad()

    #             # Update EMA model parameters
    #             ema.update()

    #             progress_bar.update(1)
    #             logs = {
    #                 "loss": loss.detach().item(),
    #                 "lr": lr_scheduler.get_last_lr()[0],
    #                 "step": global_step,
    #             }
    #             progress_bar.set_postfix(**logs)
    #             accelerator.log(logs, step=global_step)
    #             global_step += 1

    #         # After each epoch you optionally sample some demo images
    #         if accelerator.is_main_process:
    #             self.model = accelerator.unwrap_model(model)
    #             self.model.eval()

    #             # Make directory for saving images
    #             os.makedirs(os.path.join(config.output_dir, "images"), exist_ok=True)

    #             if (
    #                 epoch % config.save_image_epochs == 0
    #                 or epoch == config.num_epochs - 1
    #             ):
    #                 sample = self.sample(x[0].unsqueeze(0))
    #                 sample = sample.detach().cpu().numpy().transpose(0, 2, 3, 1)
    #                 image_path = os.path.join(
    #                     config.output_dir, "images", f"sample_{epoch}.png"
    #                 )
    #                 plt.imsave(image_path, sample[0])

    #             # Save the EMA model to HuggingFace Hub
    #             if config.push_to_hub and epoch == config.num_epochs - 1:
    #                 upload_folder(
    #                     repo_id=repo_id,
    #                     folder_path=config.output_dir,
    #                     commit_message="EMA model",
    #                 )
    #                 self.model.push_to_hub(config.hub_model_id, variant="fp16")
    #                 torch.save(ema.ema_model.module.state_dict(), "ema_model.pth")
