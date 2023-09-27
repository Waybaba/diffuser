import torch
import numpy as np

from diffuser.models.helpers import (
    extract,
    apply_conditioning,
)


@torch.no_grad()
def n_step_guided_p_sample(
    model, x, cond, t, guide, scale=0.001, t_stopgrad=0, n_guide_steps=1, scale_grad_by_std=True,
    **kwargs,
):
    model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
    model_std = torch.exp(0.5 * model_log_variance)
    model_var = torch.exp(model_log_variance)

    for _ in range(n_guide_steps):
        with torch.enable_grad():
            y, grad = guide.gradients(x, cond, t)

        if scale_grad_by_std:
            grad = model_var * grad

        grad[t < t_stopgrad] = 0 # 

        x = x + scale * grad
        x = apply_conditioning(x, cond, model.action_dim)

    model_mean, _, model_log_variance, _ = model.p_mean_variance(x=x, cond=cond, t=t)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0
    
    # ! DEBUG # add this would make the first and target step have no gradient? really? only make the first step more fixed
    model_mean = apply_conditioning(model_mean, cond, model.action_dim)
    # ! DEBUG
    return model_mean + model_std * noise, y

# def n_step_guided_p_sample_freedom_timetravel(
#     model, x, cond, t, guide, scale=0.001, t_stopgrad=0, n_guide_steps=1, scale_grad_by_std=True,
#     horizon=None, travel_interval=[0.0, 1.0], travel_repeat=1, betas=None, grad_interval=None, **kwargs
# ):
#     """
#     !!! UNFINISHED
#     travel_interval: [0.0,1.0]
    
#     travel_repeat: 1 # 1 means no travel
#     betas = []
#     horizon = 
#     """
#     if isinstance(travel_interval, str): # e.g. "[0.1,1.0]"
#         travel_interval = eval(travel_interval)
#     elif isinstance(travel_interval[0], float):
#         assert travel_interval[1] > travel_interval[0] and travel_interval[1] <= 1.0, "travel_interval should be [0.0, 1.0]"
#         travel_interval = [int(travel_interval[0]*horizon), int(travel_interval[1]*horizon)]
#     if grad_interval is not None:
#         if isinstance(grad_interval[0], float):
#             grad_interval = [int(grad_interval[0]*horizon), int(grad_interval[1]*horizon)]
    
#     model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
#     model_std = torch.exp(0.5 * model_log_variance)
#     model_var = torch.exp(model_log_variance)


#     for travel_i in range(travel_repeat):
#         model_mean, _, model_log_variance, x_recon = model.p_mean_variance(x=x, cond=cond, t=t)
#         x = x_recon.detach()
        
#         with torch.enable_grad():
#             # x0_e = (1/np.sqrt(alphas_cumprod[t])) * (x+(1-alphas_cumprod[t]))
#             y, grad = guide.gradients(x, cond, t)

#         if scale_grad_by_std:
#             grad = model_var * grad

    
#         if grad_interval is not None:
#             grad[~((grad_interval[0]<=t) & (t<grad_interval[1]))] = 0
#         else:
#             grad[t < t_stopgrad] = 0 # use this to cancel the effect of guide
        
#         # x = x + scale * grad

    
#         model_mean += scale * grad
#         x = model_mean

#         if travel_i < travel_repeat-1:
#             rand = torch.rand_like(x)
#             travel_weight_x = (1-betas[t]).sqrt()
#             travel_weight_x[~((travel_interval[0]<=t) & (t<travel_interval[1]))] = 1.0 # use this to cancel the effect of travel
#             travel_weight_noise = betas[t].sqrt()
#             travel_weight_noise[~((travel_interval[0]<=t) & (t<travel_interval[1]))] = 0.0 # use this to cancel the effect of travel
#             x = x * travel_weight_x.unsqueeze(-1).unsqueeze(-1) + travel_weight_noise.unsqueeze(-1).unsqueeze(-1) * rand

#         x = apply_conditioning(x, cond, model.action_dim)
    
#     # model_mean, _, model_log_variance, x_recon = model.p_mean_variance(x=x, cond=cond, t=t)

#     # no noise when t == 0
#     noise = torch.randn_like(x)
#     noise[t == 0] = 0
    
#     # ! DEBUG # add this would make the first and target step have no gradient? really? only make the first step more fixed
#     model_mean = apply_conditioning(model_mean, cond, model.action_dim)
#     # ! DEBUG
#     return model_mean + model_std * noise, y


def n_step_guided_p_sample_freedom_timetravel(
    model, x, cond, t, guide, scale=0.001, t_stopgrad=0, n_guide_steps=1, scale_grad_by_std=True,
    horizon=None, travel_interval=[0.0, 1.0], travel_repeat=1, betas=None, grad_interval=None, **kwargs
):
    """
    !!! UNFINISHED
    travel_interval: [0.0,1.0]
    
    travel_repeat: 1 # 1 means no travel
    betas = []
    horizon = 
    """
    if isinstance(travel_interval, str): # e.g. "[0.1,1.0]"
        travel_interval = eval(travel_interval)
    elif isinstance(travel_interval[0], float):
        assert travel_interval[1] > travel_interval[0] and travel_interval[1] <= 1.0, "travel_interval should be [0.0, 1.0]"
        travel_interval = [int(travel_interval[0]*horizon), int(travel_interval[1]*horizon)]    
    if grad_interval is not None:
        if isinstance(grad_interval[0], float):
            grad_interval = [int(grad_interval[0]*horizon), int(grad_interval[1]*horizon)]
    if t[0].item() < travel_interval[0] or t[-1].item() >= travel_interval[1]:
        travel_repeat = 1
    
    model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
    model_std = torch.exp(0.5 * model_log_variance)
    model_var = torch.exp(model_log_variance)


    for travel_i in range(travel_repeat):
        model_mean, _, model_log_variance, x_recon = model.p_mean_variance(x=x, cond=cond, t=t)
        model_std = torch.exp(0.5 * model_log_variance)
        model_var = torch.exp(model_log_variance)
        
        with torch.enable_grad():
            y, grad = guide.gradients(x_recon.detach(), cond, t)

        if scale_grad_by_std:
            grad = model_var * grad
        else:
            # grad: N, T, obs_dim
            # model_std: N, 1, 1
            # norm grad to have the same vector length as model_std
            grad_mean = (grad*grad).mean(dim=[1,2], keepdim=True).sqrt()
            grad = grad / grad_mean
            grad = (1-torch.cumprod(betas,dim=0)[t]).sqrt().unsqueeze(-1).unsqueeze(-1) * grad
            
        if grad_interval is not None:
            grad[~((grad_interval[0]<=t) & (t<grad_interval[1]))] = 0
        
        # ! DEBUG # add this would make the first and target step have no gradient? really? only make the first step more fixed
        model_mean = apply_conditioning(model_mean, cond, model.action_dim)
        noise = torch.randn_like(x)
        noise[t == 0] = 0
        x = model_mean + model_std * noise
        x = x + scale * grad
        x = apply_conditioning(x, cond, model.action_dim)

        if travel_i < travel_repeat-1:
            rand = torch.rand_like(x)
            travel_weight_x = (1-betas[t]).sqrt()
            travel_weight_x[~((travel_interval[0]<=t) & (t<travel_interval[1]))] = 1.0 # use this to cancel the effect of travel
            travel_weight_noise = betas[t].sqrt()
            travel_weight_noise[~((travel_interval[0]<=t) & (t<travel_interval[1]))] = 0.0 # use this to cancel the effect of travel
            x = x * travel_weight_x.unsqueeze(-1).unsqueeze(-1) + travel_weight_noise.unsqueeze(-1).unsqueeze(-1) * rand

    # model_mean, _, model_log_variance, x_recon = model.p_mean_variance(x=x, cond=cond, t=t)

    # no noise when t == 0
    # ! DEBUG
    return x, y