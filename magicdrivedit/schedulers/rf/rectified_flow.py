from typing import List
import logging

import torch
from torch.distributions import LogisticNormal
from einops import rearrange
from structured_noise import generate_structured_noise_batch_vectorized

# some code are inspired by https://github.com/magic-research/piecewise-rectified-flow/blob/main/scripts/train_perflow.py
# and https://github.com/magic-research/piecewise-rectified-flow/blob/main/src/scheduler_perflow.py


def mean_flat(tensor: torch.Tensor, mask=None):
    """
    Take the mean over all non-batch dimensions.
    """
    if mask is None:
        return tensor.mean(dim=list(range(1, len(tensor.shape))))
    else:
        assert tensor.dim() == 5
        assert tensor.shape[2] == mask.shape[1]
        tensor = rearrange(tensor, "b c t h w -> b t (c h w)")
        denom = mask.sum(dim=1) * tensor.shape[-1]
        loss = (tensor * mask.unsqueeze(2)).sum(dim=1).sum(dim=1) / denom
        return loss


def _extract_into_tensor(arr: torch.Tensor, timesteps: torch.Tensor, broadcast_shape: List[int]):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = arr.to(timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)


def timestep_transform(
    t,
    model_kwargs,
    base_resolution=512 * 512,
    base_num_frames=1,
    scale=1.0,
    num_timesteps=1,
    cog_style=False,
):
    # Force fp16 input to fp32 to avoid nan output
    for key in ["height", "width", "num_frames"]:
        if model_kwargs[key].dtype == torch.float16:
            model_kwargs[key] = model_kwargs[key].float()

    t = t / num_timesteps
    resolution = model_kwargs["height"] * model_kwargs["width"]
    ratio_space = (resolution / base_resolution).sqrt()
    # NOTE: currently, we do not take fps into account
    # NOTE: temporal_reduction is hardcoded, this should be equal to the temporal reduction factor of the vae
    # TODO: hard-coded, may change later!
    if model_kwargs["num_frames"][0] == 1:
        num_frames = torch.ones_like(model_kwargs["num_frames"])
    else:
        if cog_style:
            num_frames = model_kwargs["num_frames"] // 4 + model_kwargs["num_frames"] % 2
        else:
            num_frames = model_kwargs["num_frames"] // 17 * 5
    assert (num_frames >= 1).all(), "num_frames cannot be less than 1"
    ratio_time = (num_frames / base_num_frames).sqrt()

    ratio = ratio_space * ratio_time * scale
    assert (ratio > 0).all(), "ratio cannot be 0"
    new_t = ratio * t / (1 + (ratio - 1) * t)

    new_t = new_t * num_timesteps
    return new_t


class RFlowScheduler:
    def __init__(
        self,
        num_timesteps=1000,
        num_sampling_steps=10,
        use_discrete_timesteps=False,
        sample_method="uniform",
        loc=0.0,
        scale=1.0,
        use_timestep_transform=False,
        transform_scale=1.0,
        cog_style_trans=False,
    ):
        self.num_timesteps = num_timesteps
        self.num_sampling_steps = num_sampling_steps
        self.use_discrete_timesteps = use_discrete_timesteps

        # sample method
        assert sample_method in ["uniform", "logit-normal"]
        assert (
            sample_method == "uniform" or not use_discrete_timesteps
        ), "Only uniform sampling is supported for discrete timesteps"
        self.sample_method = sample_method
        if sample_method == "logit-normal":
            self.distribution = LogisticNormal(torch.tensor([loc]), torch.tensor([scale]))
            self.sample_t = lambda x: self.distribution.sample((x.shape[0],))[:, 0].to(x.device)

        # timestep transform
        self.use_timestep_transform = use_timestep_transform
        self.transform_scale = transform_scale
        if cog_style_trans:
            logging.warning("Use `cog_style_trans`. Please make sure train&inference is consistent!")
        self.cog_style_trans = cog_style_trans

    def training_losses(self, model, x_start, model_kwargs=None, noise=None, mask=None, weights=None, t=None):
        """
        Compute training losses for a single timestep.
        Arguments format copied from magicdrivedit/schedulers/iddpm/gaussian_diffusion.py/training_losses
        Note: t is int tensor and should be rescaled from [0, num_timesteps-1] to [1,0]
        """
        if t is None:
            if self.use_discrete_timesteps:
                t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device)
            elif self.sample_method == "uniform":
                t = torch.rand((x_start.shape[0],), device=x_start.device) * self.num_timesteps
            elif self.sample_method == "logit-normal":
                t = self.sample_t(x_start) * self.num_timesteps

            if self.use_timestep_transform:
                t = timestep_transform(t, model_kwargs, scale=self.transform_scale, num_timesteps=self.num_timesteps, cog_style=self.cog_style_trans)

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape

        x_t = self.add_noise(x_start, noise, t)
        if mask is not None:
            t0 = torch.zeros_like(t)
            x_t0 = self.add_noise(x_start, noise, t0)
            x_t = torch.where(mask[:, None, :, None, None], x_t, x_t0)

        terms = {}
        model_output = model(x_t, t, **model_kwargs)
        if model_output.shape[1] == 2 * x_t.shape[1]:
            model_output = model_output.chunk(2, dim=1)[0]
        velocity_pred = model_output
        if weights is None:
            loss = mean_flat((velocity_pred - (x_start - noise)).pow(2), mask=mask)
        else:
            weight = _extract_into_tensor(weights, t, x_start.shape)
            loss = mean_flat(weight * (velocity_pred - (x_start - noise)).pow(2), mask=mask)
        terms["loss"] = loss

        return terms

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        compatible with diffusers add_noise()
        """
        timepoints = timesteps.float() / self.num_timesteps
        timepoints = 1 - timepoints  # [1,1/1000]

        # timepoint  (bsz) noise: (bsz, 4, frame, w ,h)
        # expand timepoint to noise shape
        timepoints = timepoints.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        timepoints = timepoints.repeat(1, noise.shape[1], noise.shape[2], noise.shape[3], noise.shape[4])

        return timepoints * original_samples + (1 - timepoints) * noise

class RFlowSchedulerPPD(RFlowScheduler):
    
    def training_losses(self, model, x_start, model_kwargs=None, noise=None, mask=None, weights=None, t=None):
        """
        Compute training losses for a single timestep.
        Arguments format copied from magicdrivedit/schedulers/iddpm/gaussian_diffusion.py/training_losses
        Note: t is int tensor and should be rescaled from [0, num_timesteps-1] to [1,0]
        """
        if t is None:
            if self.use_discrete_timesteps:
                t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device)
            elif self.sample_method == "uniform":
                t = torch.rand((x_start.shape[0],), device=x_start.device) * self.num_timesteps
            elif self.sample_method == "logit-normal":
                t = self.sample_t(x_start) * self.num_timesteps

            if self.use_timestep_transform:
                t = timestep_transform(t, model_kwargs, scale=self.transform_scale, num_timesteps=self.num_timesteps, cog_style=self.cog_style_trans)

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            B, NC_C, T, H, W = x_start.shape
            # Reshape to (B*NC*T, C, H, W) for 2D processing
            x_flat = rearrange(x_start, "b (c nc) t h w -> (b nc t) c h w", nc = 6)# TODO: hard-coded nc=6
            input_noise = torch.randn_like(x_flat)
            r0 = min(H, W) // 4
            u = torch.rand(B, device=x_flat.device)
            cutoff_radius = r0 + (-torch.log(u) / 0.1)
            chunk_size = 6
            frames_per_scene = chunk_size * T
            structured_noise_list = []
            for i in range(0, x_flat.shape[0], chunk_size):
                scene_idx = i // frames_per_scene
                x_chunk = x_flat[i : i + chunk_size]
                
                noise_chunk = input_noise[i : i + chunk_size]
                out_chunk = generate_structured_noise_batch_vectorized(
                    x_chunk,
                    cutoff_radius=cutoff_radius[scene_idx].item(),
                    transition_width=2.0,
                    input_noise=noise_chunk,
                )
                structured_noise_list.append(out_chunk)

                structured_noise_flat = torch.cat(structured_noise_list, dim=0)
                structured_noise_flat = structured_noise_flat.to(
                    device=x_flat.device, dtype=x_flat.dtype
                )
            noise = rearrange(
                structured_noise_flat,
                "(b nc t) c h w -> b (c nc) t h w",
                b=B,
                nc=6,
                t=T,
            )
        assert noise.shape == x_start.shape

        x_t = self.add_noise(x_start, noise, t)
        if mask is not None:
            t0 = torch.zeros_like(t)
            x_t0 = self.add_noise(x_start, noise, t0)
            x_t = torch.where(mask[:, None, :, None, None], x_t, x_t0)

        terms = {}
        model_output = model(x_t, t, **model_kwargs)
        if model_output.shape[1] == 2 * x_t.shape[1]:
            model_output = model_output.chunk(2, dim=1)[0]
        velocity_pred = model_output
        if weights is None:
            loss = mean_flat((velocity_pred - (x_start - noise)).pow(2), mask=mask)
        else:
            weight = _extract_into_tensor(weights, t, x_start.shape)
            loss = mean_flat(weight * (velocity_pred - (x_start - noise)).pow(2), mask=mask)
        terms["loss"] = loss

        return terms
class RFlowSchedulerBrushNet(RFlowScheduler):

    def training_losses(
        self,
        model,
        x_start,
        x_inpaint,
        mask_inpaint,
        model_kwargs=None,
        noise=None,
        mask=None,
        weights=None,
        t=None,
    ):
        """
        Compute training losses for a single timestep.
        Arguments format copied from magicdrivedit/schedulers/iddpm/gaussian_diffusion.py/training_losses
        Note: t is int tensor and should be rescaled from [0, num_timesteps-1] to [1,0]
        """
        if t is None:
            if self.use_discrete_timesteps:
                t = torch.randint(
                    0, self.num_timesteps, (x_start.shape[0],), device=x_start.device
                )
            elif self.sample_method == "uniform":
                t = (
                    torch.rand((x_start.shape[0],), device=x_start.device)
                    * self.num_timesteps
                )
            elif self.sample_method == "logit-normal":
                t = self.sample_t(x_start) * self.num_timesteps

            if self.use_timestep_transform:
                t = timestep_transform(
                    t,
                    model_kwargs,
                    scale=self.transform_scale,
                    num_timesteps=self.num_timesteps,
                    cog_style=self.cog_style_trans,
                )

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape

        x_t = self.add_noise(x_start, noise, t)
        if mask is not None:
            t0 = torch.zeros_like(t)
            x_t0 = self.add_noise(x_start, noise, t0)
            x_t = torch.where(mask[:, None, :, None, None], x_t, x_t0)

        terms = {}
        model_output = model(x_t, x_inpaint, mask_inpaint, t, **model_kwargs)
        if model_output.shape[1] == 2 * x_t.shape[1]:
            model_output = model_output.chunk(2, dim=1)[0]
        velocity_pred = model_output
        if weights is None:
            loss = mean_flat((velocity_pred - (x_start - noise)).pow(2), mask=mask)
        else:
            weight = _extract_into_tensor(weights, t, x_start.shape)
            loss = mean_flat(
                weight * (velocity_pred - (x_start - noise)).pow(2), mask=mask
            )
        terms["loss"] = loss

        return terms


class RFlowSchedulerSDEBrushNet(RFlowScheduler):

    def training_losses(
        self,
        model,
        x_start,
        x_inpaint,
        mask_inpaint,
        model_kwargs=None,
        noise=None,
        mask=None,
        weights=None,
        t=None,
        t_inpaint=None,
    ):
        """
        Compute training losses for a single timestep.
        Arguments format copied from magicdrivedit/schedulers/iddpm/gaussian_diffusion.py/training_losses
        Note: t is int tensor and should be rescaled from [0, num_timesteps-1] to [1,0]
        
        Args:
            t_inpaint: Independent timestep for x_inpaint. If None, will be sampled independently.
        """
        if t is None:
            if self.use_discrete_timesteps:
                t = torch.randint(
                    0, self.num_timesteps, (x_start.shape[0],), device=x_start.device
                )
            elif self.sample_method == "uniform":
                t = (
                    torch.rand((x_start.shape[0],), device=x_start.device)
                    * self.num_timesteps
                )
            elif self.sample_method == "logit-normal":
                t = self.sample_t(x_start) * self.num_timesteps

            if self.use_timestep_transform:
                t = timestep_transform(
                    t,
                    model_kwargs,
                    scale=self.transform_scale,
                    num_timesteps=self.num_timesteps,
                    cog_style=self.cog_style_trans,
                )
        # Sample independent timestep for x_inpaint
        if t_inpaint is None:
            t_inpaint = (
                torch.rand((x_start.shape[0],), device=x_start.device) ** 0.2
                * self.num_timesteps
            )
            if self.use_timestep_transform:
                t_inpaint = timestep_transform(
                    t_inpaint,
                    model_kwargs,
                    scale=self.transform_scale,
                    num_timesteps=self.num_timesteps,
                    cog_style=self.cog_style_trans,
                )

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape

        x_t = self.add_noise(x_start, noise, t)
        if mask is not None:
            t0 = torch.zeros_like(t)
            x_t0 = self.add_noise(x_start, noise, t0)
            x_t = torch.where(mask[:, None, :, None, None], x_t, x_t0)

        terms = {}
        model_output = model(x_t, x_inpaint, mask_inpaint, t, t_inpaint, num_timesteps=self.num_timesteps, **model_kwargs)
        if model_output.shape[1] == 2 * x_t.shape[1]:
            model_output = model_output.chunk(2, dim=1)[0]
        velocity_pred = model_output
        if weights is None:
            loss = mean_flat((velocity_pred - (x_start - noise)).pow(2), mask=mask)
        else:
            weight = _extract_into_tensor(weights, t, x_start.shape)
            loss = mean_flat(
                weight * (velocity_pred - (x_start - noise)).pow(2), mask=mask
            )
        terms["loss"] = loss

        return terms
