import numpy as np
import torch
from torch.optim.optimizer import Optimizer

class AnchorAdamW(Optimizer):
    def __init__(self, params, lr = 5e-4, betas = (0., 0.999), eps = 1e-5, weight_decay = 0., mixing_anchor = False,
                 averaging_anchor = True):
        # Params should be an iterable of parameters or param groups
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(lr = lr, betas = betas, eps = eps, weight_decay = weight_decay, mixing_anchor = mixing_anchor,
                        averaging_anchor = averaging_anchor)
        super(AnchorAdamW, self).__init__(params, defaults)

    def reset_anchor(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) != 0:
                    # state['anchor'] = torch.clone(torch.detach(p))
                    state['anchor'] = state['anchor_']
                    state['anchor_'] = torch.clone(torch.detach(p))

                    state['counter'] = state['counter_']
                    state['counter_'] = 0

    def step(self, closure = None):
        # closure is an optional function that re-evaluates the model and returns loss
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('This AdamW does not support sparse gradients.')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format = torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format = torch.preserve_format)
                    # Anchor
                    state['anchor'] = torch.clone(torch.detach(p))
                    state['anchor_'] = torch.clone(torch.detach(p))

                    state['counter'] = 1
                    state['counter_'] = 0

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                step = state['step']

                state['counter_'] += 1

                # Update first and second moment running averages
                exp_avg.mul_(beta1).add_(grad, alpha = 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value = 1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                # Compute bias-corrected estimates
                denom = (exp_avg_sq.sqrt() / np.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1

                # Apply decoupled weight decay
                # This ensures weight decay is applied directly to parameters
                if group['weight_decay'] != 0:
                    with torch.no_grad():
                        wd = group['lr'] * group['weight_decay']
                        p.mul_(1 - wd)
                        # p.add_(state['anchor'], alpha = wd)

                        if group['mixing_anchor']:
                            p.add_(state['anchor_'], alpha = wd * state['counter_'] / state['counter'])
                            p.add_(state['anchor'], alpha = wd * (1 - state['counter_'] / state['counter']))
                        else:
                            if group['averaging_anchor']:
                                state['anchor_'].mul_(1 - 2 / (state['counter_'] + 1)).add(p, alpha = 2 / (
                                        state['counter_'] + 1))
                            p.add_(state['anchor'], alpha = wd)

                # Parameter update
                with torch.no_grad():
                    p.addcdiv_(exp_avg, denom, value = -step_size)

        return loss