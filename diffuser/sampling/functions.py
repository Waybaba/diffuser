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

    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0
    
    # ! DEBUG # add this would make the first and target step have no gradient? really? only make the first step more fixed
    model_mean = apply_conditioning(model_mean, cond, model.action_dim)
    # ! DEBUG
    return model_mean + model_std * noise, y

def n_step_guided_p_sample_freedom_timetravel(
    model, x, cond, t, guide, scale=0.001, t_stopgrad=0, n_guide_steps=1, scale_grad_by_std=True,
    horizon=None, travel_interval=[0.0, 1.0], travel_repeat=1, alphas_cumprod=None, **kwargs
):
    """
    !!! UNFINISHED
    travel_interval: [0.0,1.0]
    travel_repeat: 1 # 1 means no travel
    alphas_cumprod = []
    horizon = 
    """
    model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
    model_std = torch.exp(0.5 * model_log_variance)
    model_var = torch.exp(model_log_variance)
    if isinstance(travel_interval[0], float):
        assert travel_interval[1] > travel_interval[0] and travel_interval[1] <= 1.0, "travel_interval should be [0.0, 1.0]"
        travel_interval = [int(travel_interval[0]*horizon), int(travel_interval[1]*horizon)]
    if t in range(travel_interval[0], travel_interval[1]): # travel
        alphas_cumprod[t] = 1.0
    
    for travel_i in range(travel_repeat):
        with torch.enable_grad():
            # x0_e = (1/np.sqrt(alphas_cumprod[t])) * (x+(1-alphas_cumprod[t]))
            y, grad = guide.gradients(x, cond, t)

        if scale_grad_by_std:
            grad = model_var * grad

        grad[t < t_stopgrad] = 0 # use this to cancel the effect of guide
    
        x = x + scale * grad

        if travel_i < travel_repeat-1:
            rand = torch.rand_like(x)
            travel_weight = alphas_cumprod[t].sqrt()
            travel_weight[~(travel_interval[0]<=t<travel_interval[1])] = 1.0 # use this to cancel the effect of travel
            x = x * alphas_cumprod[t].sqrt() + (1-alphas_cumprod[t]).sqrt() * rand
            
        x = apply_conditioning(x, cond, model.action_dim)

    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0
    
    # ! DEBUG # add this would make the first and target step have no gradient? really? only make the first step more fixed
    model_mean = apply_conditioning(model_mean, cond, model.action_dim)
    # ! DEBUG
    return model_mean + model_std * noise, y