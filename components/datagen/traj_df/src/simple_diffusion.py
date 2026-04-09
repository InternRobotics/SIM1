"""
This file contains an implementation of the pseudocode from the paper
"Simple Diffusion: End-to-End Diffusion for High Resolution Images"
by Emiel Hoogeboom, Tim Salimans, and Jonathan Ho.

Reference:
Hoogeboom, E., Salimans, T., & Ho, J. (2023).
Simple Diffusion: End-to-End Diffusion for High Resolution Images.
Retrieved from https://arxiv.org/abs/2301.11093
"""

import math
from typing import Optional
import os
import torch
import torch.nn as nn
from torch.special import expm1
import numpy as np


from collections import namedtuple
from .models.utils import linear_beta_schedule, cosine_beta_schedule, sigmoid_beta_schedule, extract, EinopsWrapper
from torch.amp import autocast
from torch.nn import functional as F
# helper
def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))

ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start", "model_out"])


class simpleDiffusion(nn.Module):
    def __init__(
        self,
        model,
        noise_size=64,
        pred_param="v",
        schedule="shifted_cosine",
        steps=32,
        uncertainty_scale=1.0,
        condition_time=1,
        scheduling_matrix="full_sequence",
        final_step=True,
        fill_scene_tensor=True,
        device=None,
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
        self.image_d = 128

        # Model
        assert isinstance(
            model, nn.Module
        ), "Model must be an instance of torch.nn.Module."
        self.model = model

        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Number of parameters: {num_params}")

        # Steps
        self.steps = steps
        self.uncertainty_scale = uncertainty_scale
        self.condition_time = condition_time
        self.scheduling_matrix = scheduling_matrix
        self.final_step = final_step
        self.fill_scene_tensor = fill_scene_tensor
        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = torch.device(device)

    def logsnr_schedule_cosine(self, t, logsnr_min=-15, logsnr_max=15):
        """
        Function to compute the logSNR schedule at timepoint t with cosine:

        logSNR(t) = -2 * log (tan (pi * t / 2))


        logsnr_t = -2 * log(tan(t_min + t * (t_max - t_min)))

        Args:
        t (int): The timepoint t.
        logsnr_min (int): The minimum logSNR value.
        logsnr_max (int): The maximum logSNR value.

        Returns:
        logsnr_t (float): The logSNR value at timepoint t.
        """
        t = torch.as_tensor(t, device=self.device, dtype=torch.float32)

        logsnr_max = logsnr_max + math.log(self.noise_d / self.image_d)
        logsnr_min = logsnr_min + math.log(self.noise_d / self.image_d)
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))

        # logsnr_t = -2 * log(torch.tan(torch.tensor(t_min + t * (t_max - t_min))))
        logsnr_t = -2 * log(torch.tan((t_min + t * (t_max - t_min)).clone().detach()))

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

    def clip(self, x):
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
        recip_alpha_t = torch.sqrt(1.0 / torch.sigmoid(logsnr_t))
        recipml_alpha_t = torch.sqrt(1.0 / torch.sigmoid(logsnr_t) - 1.0)

        if self.pred_param == "v":
            x_pred = alpha_t * z_t - sigma_t * pred # pred_x_start
            pred_noise = (recip_alpha_t * z_t - x_pred) / recipml_alpha_t # pred_noise
        elif self.pred_param == "eps":
            x_pred = (z_t - sigma_t * pred) / alpha_t

        mu = alpha_s * (z_t * (1 - c) / alpha_t + c * x_pred)
        variance = (sigma_s**2) * c

        return mu, variance

    def compute_start_from_v(self, z_t, pred, logsnr_t, logsnr_s):
        c = -expm1(logsnr_t - logsnr_s)
        alpha_t = torch.sqrt(torch.sigmoid(logsnr_t))
        alpha_s = torch.sqrt(torch.sigmoid(logsnr_s))
        sigma_t = torch.sqrt(torch.sigmoid(-logsnr_t))
        sigma_s = torch.sqrt(torch.sigmoid(-logsnr_s))
        recip_alpha_t = torch.sqrt(1.0 / torch.sigmoid(logsnr_t))
        recipml_alpha_t = torch.sqrt(1.0 / torch.sigmoid(logsnr_t) - 1.0)

        if self.pred_param == "v":
            x_pred = alpha_t * z_t - sigma_t * pred # pred_x_start
        return x_pred

    def _generate_scheduling_matrix(self, scene_tensor, zero_init=True):
        B, NA, NT = scene_tensor.shape[:3]
        if self.scheduling_matrix == "pyramid":
            return self._generate_pyramid_scheduling_matrix(zero_init, NT, self.uncertainty_scale)[:,np.newaxis,np.newaxis,:].repeat(B, 1).repeat(NA, 2)
        elif self.scheduling_matrix == "full_sequence":
            scheduling_matrix = np.linspace(1, 0, self.steps)[:,np.newaxis,np.newaxis,np.newaxis].repeat(B, 1).repeat(NA, 2).repeat(NT, 3)
            if zero_init:
                scheduling_matrix[:,:,:,:self.condition_time] = 0 #[m B NA NT]
            return scheduling_matrix.astype(np.float32)           
        elif self.scheduling_matrix == "autoregressive":
            return self._generate_pyramid_scheduling_matrix(zero_init, NT, self.steps)[:,np.newaxis,np.newaxis,:].repeat(B, 1).repeat(NA, 2)
        elif self.scheduling_matrix == "trapezoid":
            return self._generate_trapezoid_scheduling_matrix(zero_init, NT, self.uncertainty_scale)[:,np.newaxis,np.newaxis,:].repeat(B, 1).repeat(NA, 2)

    def _generate_pyramid_scheduling_matrix(self, zero_init: bool, horizon: int, uncertainty_scale: float):
        if zero_init:
            horizon = horizon - self.condition_time
        height = self.steps + int((horizon - 1) * uncertainty_scale) + 1
        scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
        for m in range(height):
            for t in range(horizon):
                scheduling_matrix[m, t] = self.steps + int(t * uncertainty_scale) - m
        scheduling_matrix = np.concatenate([np.zeros((height, self.condition_time)), scheduling_matrix], axis=1)
        scheduling_matrix = np.clip(scheduling_matrix, 0, self.steps)/self.steps

        return scheduling_matrix.astype(np.float32)

    def _generate_trapezoid_scheduling_matrix(self, zero_init: bool, horizon: int, uncertainty_scale: float):
        if zero_init:
            horizon = horizon - self.condition_time
        extra_step = (horizon+1) % 2
        height = self.steps + int((horizon + 1) // 2 * uncertainty_scale) + extra_step
        scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
        for m in range(height):
            for t in range((horizon + 1) // 2 + extra_step):
                scheduling_matrix[m, t] = self.steps + int(t * uncertainty_scale) - m
                scheduling_matrix[m, -t] = self.steps + int(t * uncertainty_scale) - m

        scheduling_matrix = np.concatenate([np.zeros((height, self.condition_time)), scheduling_matrix], axis=1)
        scheduling_matrix = np.clip(scheduling_matrix, 0, self.steps)/self.steps

        return scheduling_matrix.astype(np.float32)

    def _filling_scene_tensor(self, scene_tensor, z_t, keep_mask, scaling_matrix):
        if not self.fill_scene_tensor:
            return scaling_matrix, z_t
        # keep_mask: [B NA NT D], scaling_matrix: [m B NA NT]
        z_t = torch.where(keep_mask.bool(), scene_tensor, z_t)
        keep_mask = keep_mask[:,:,:,0].bool().unsqueeze(0).repeat(scaling_matrix.shape[0], 1, 1, 1)
        scaling_matrix = np.where(~keep_mask.cpu().numpy(), scaling_matrix, np.full(scaling_matrix.shape, 0).astype(np.float32))
        return scaling_matrix, z_t


    def sampler_step(self, scene_tensor, x, keep_mask, u_t, u_s, local_context, global_context, valid_mask, schedule_func, compute_mu):

        orig_x = x.clone().detach()
        logsnr_t = schedule_func(u_t)
        logsnr_t = logsnr_t.clone().detach().to(scene_tensor.device)
        alpha_t = (
            torch.sqrt(torch.sigmoid(logsnr_t))
            .view(scene_tensor.shape[0], scene_tensor.shape[1], scene_tensor.shape[2], 1)
            .to(scene_tensor.device)
        )
        sigma_t = (
            torch.sqrt(torch.sigmoid(-logsnr_t))
            .view(scene_tensor.shape[0], scene_tensor.shape[1], scene_tensor.shape[2], 1)
            .to(scene_tensor.device)
        )
        scaled_context, eps_t = self.diffuse(x, alpha_t, sigma_t)
        z_t = torch.where(keep_mask.bool(), scaled_context, orig_x)

        logsnr_s = schedule_func(u_s)
        logsnr_s = logsnr_s.to(scene_tensor.device)

        u_t = torch.tensor(u_t, device=x.device)
        u_s = torch.tensor(u_s, device=x.device)

        pred = self.model(
            local_context=local_context,
            diffused_scene_tensor=z_t,
            valid_mask=valid_mask,
            diffusion_times=u_t,
            global_context=global_context,
        )
        mu, variance = self.ddpm_sampler_step(
            z_t, pred, logsnr_t.unsqueeze(-1).repeat(1, 1, 1, scene_tensor.shape[3]), logsnr_s.unsqueeze(-1).repeat(1, 1, 1, scene_tensor.shape[3])
        )

        intermidiates = mu.clone().detach()
        if not compute_mu:
            return mu, intermidiates

        # apply keep_mask
        mu[keep_mask.bool()] = scene_tensor[keep_mask.bool()]
        z_t = mu + torch.randn_like(mu) * torch.sqrt(variance)

        return z_t, intermidiates

    @torch.no_grad()
    def sample(
        self,
        batch,
        z_t: Optional[torch.Tensor] = None,
        return_intermidates: bool = False,
        use_guidance_fn: bool = False,
    ):
        """
        Standard DDPM sampling procedure. Begun by sampling z_T ~ N(0, 1)
        and then repeatedly sampling z_s ~ p(z_s | z_t)

        Args:
        x_shape (tuple): The shape of the input tensor.
        global_context (torch.Tensor): The global context tensor.


        Returns:
        x_pred (torch.Tensor): The predicted tensor.
        """
        scene_tensor, valid_mask, global_context, diffusion_times, keep_mask = self.map_batch_to_diffusion_inputs(batch, self.device)

        if z_t is None:
            z_t = torch.randn(scene_tensor.shape).to(scene_tensor.device)
        fuse_mask = False

        if fuse_mask == True:
            keep_mask = keep_mask * valid_mask.unsqueeze(-1)
        local_context = scene_tensor * keep_mask
        # add the valid mask as a channel
        local_context = torch.cat([local_context, keep_mask], dim=-1)

        intermidiates = []
        schedule_func = (
            self.logsnr_schedule_cosine
            if self.schedule == "cosine"
            else self.logsnr_schedule_cosine_shifted
        )
        
        # Steps T -> 1
        scaling_matrix = self._generate_scheduling_matrix(scene_tensor)
        scaling_matrix, z_t = self._filling_scene_tensor(scene_tensor, z_t, keep_mask, scaling_matrix)
        gt_replace = os.environ.get('GT_REPLACE', False)
        if gt_replace:
            res = torch.zeros_like(scene_tensor)
            gt_replace_mask = np.logical_not(scaling_matrix[0].astype(bool))
            res[gt_replace_mask] = scene_tensor[gt_replace_mask] 
            if self.fill_scene_tensor:
                res = torch.where(keep_mask.bool(), scene_tensor, res)

        # original_z_t = z_t.clone()
        for t in range(scaling_matrix.shape[0]-1):
            u_t = scaling_matrix[t]
            u_s = scaling_matrix[t+1]
            z_t, mu = self.sampler_step(scene_tensor, z_t, keep_mask, u_t, u_s, local_context, global_context, valid_mask, schedule_func, True)
            # z_t = torch.where(keep_mask.bool(), original_z_t, z_t)
            if return_intermidates:
                intermidiates.append(mu)
            if gt_replace:
                gt_replace_mask = np.logical_and(np.logical_not(u_s.astype(bool)), u_t.astype(bool))
                res[gt_replace_mask] = z_t[gt_replace_mask]
                z_t[gt_replace_mask] = scene_tensor[gt_replace_mask]
        if gt_replace:
            z_t = res

        if self.final_step:
            # Final step
            u_t = np.full_like(u_t, 1 / self.steps)
            u_s = np.full_like(u_s, 0)

            x_pred, _ = self.sampler_step(scene_tensor, z_t, keep_mask, u_t, u_s, local_context, global_context, valid_mask, schedule_func, False)

        x_pred[keep_mask.bool()] = scene_tensor[keep_mask.bool()]
        if return_intermidates:
            intermidiates.append(x_pred)
        return x_pred, intermidiates
    
    def compute_loss(
        self,
        batch
    ):
        """
        A function to compute the loss of the model. The loss is computed as the mean squared error
        between the predicted noise tensor and the true noise tensor. Various prediction parameterizations
        imply various weighting schemes as outlined in Kingma et al. (2023)

        Returns:
        loss (torch.Tensor): The loss value.
        """
        scene_tensor, valid_mask, global_context, diffusion_times, control_mask = self.map_batch_to_diffusion_inputs(batch, self.device)


        schedule_func = (
            self.logsnr_schedule_cosine
            if self.schedule == "cosine"
            else self.logsnr_schedule_cosine_shifted
        )
        task_mask = control_mask
        # if self.fill_scene_tensor:
        #     keep_mask = task_mask[:,:,:,0].bool()
        #     diffusion_times = torch.where(~keep_mask, diffusion_times, torch.full(diffusion_times.shape, 0.017, device=diffusion_times.device))
        logsnr_t = schedule_func(diffusion_times)
        logsnr_t = logsnr_t.to(scene_tensor.device)
        alpha_t = (
            torch.sqrt(torch.sigmoid(logsnr_t))
            .view(scene_tensor.shape[0], scene_tensor.shape[1], scene_tensor.shape[2], 1)
            .to(scene_tensor.device)
        )
        sigma_t = (
            torch.sqrt(torch.sigmoid(-logsnr_t))
            .view(scene_tensor.shape[0], scene_tensor.shape[1], scene_tensor.shape[2], 1)
            .to(scene_tensor.device)
        )
        z_t, eps_t = self.diffuse(scene_tensor, alpha_t, sigma_t)
        # create the local context
        local_context = scene_tensor * task_mask
        # add the valid mask as a channel
        local_context = torch.cat([local_context, task_mask], dim=-1)

        pred = self.model(
            local_context=local_context,
            diffused_scene_tensor=z_t,
            valid_mask=valid_mask,
            diffusion_times=diffusion_times,
            global_context=global_context,
        )

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

        weight = weight.view(scene_tensor.shape[0], scene_tensor.shape[1], scene_tensor.shape[2], 1)

        # add zero weights to invalid pixels
        weight = weight * valid_mask.unsqueeze(-1)
        # weight = weight * torch.logical_not(task_mask)

        loss = torch.sum(weight * (eps_pred - eps_t) ** 2)

        return (
            loss,
            dict(loss_flow=loss.item()),
        )

    def map_batch_to_diffusion_inputs(self, batch, device):
        """
        Convert raw dataloader batch into diffusion-ready inputs.

        Inputs:
            history: [B, T, ND]
            src:     [B, ND]
            tgt:     [B, ND]
            x1:      [B, NT, ND]
            mask:    [B, NT]
            lengths: [B]

        Outputs:
            scene_tensor:  [B, 1, NT, ND]
            valid_mask:    [B, 1, NT]
            global_context:[B, T, ND]
            diffusion_times:[B, NT]
            control_mask:  [B, 1, NT, ND]
        """

        history = batch["history"].to(device)   # (B,T,ND)
        src = batch["src"].to(device)           # (B,ND)
        tgt = batch["tgt"].to(device)           # (B,ND)
        x1 = batch["traj"].to(device)           # (B,NT,ND)
        mask = batch["mask"].to(device)         # (B,NT)
        lengths = batch["lengths"].to(device)   # (B,)

        B, NT, ND = x1.shape

        # (1) scene_tensor = x1 with NA = 1
        scene_tensor = x1.unsqueeze(1)          # (B,1,NT,ND)

        # (2) valid_mask = mask with NA = 1
        valid_mask = mask.unsqueeze(1)          # (B,1,NT)

        # (3) global context = history
        global_context = history                # (B,T,ND)

        # (4) random diffusion times ∈ [0,1]
        diffusion_times = torch.rand(B, NT, device=device)

        # (5) control mask: only first and last valid index = 1
        control_mask = torch.zeros((B, 1, NT, ND), device=device)

        for b in range(B):
            l = lengths[b].item()               # actual length
            control_mask[b, 0, 0] = 1      # first control → src
            control_mask[b, 0, l-1] = 1    # last control → tgt
        
        return scene_tensor, valid_mask, global_context, diffusion_times, control_mask

    def train_step(
        self,
        batch
    ):
        return self.compute_loss(batch)

    def infer_step(
        self,
        batch
    ):
        return self.sample(batch)[0].squeeze(), batch["mask"].to(self.device)

    def save(self, path):
        torch.save({'model': self.model.state_dict(),}, path)

    def load(self, path, map_location=None):
        ck = torch.load(path, map_location=map_location)
        state_dict = ck['model']

        # If checkpoint was saved from DDP, strip leading "module." prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        # Load state dict with strict=False to ignore missing / unexpected keys
        missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)

        if missing:
            print(f"[Warning] Missing keys when loading: {missing}")
        if unexpected:
            print(f"[Warning] Unexpected keys when loading: {unexpected}")

        if 'optim' in ck and hasattr(self, 'optim'):
            self.optim.load_state_dict(ck['optim'])
