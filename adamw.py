from torch.optim import Optimizer
import torch

class AdamW(Optimizer):

    def __init__(self, params, lr: float=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self):
        for param_group in self.param_groups:
           lr, eps = param_group["lr"], param_group["eps"]
           beta_1, beta_2 = param_group["betas"]
           wd = param_group["weight_decay"]

           for p in param_group["params"]:
                
                if p.grad is None:
                    continue
                # apply weight decay only for the weights and the embeddings
                if wd != 0 and p.ndim >= 2:
                    p.mul_(1 - lr * wd)

                g = p.grad.detach()
                g32 = g.to(torch.float32)
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p, dtype=torch.float32)
                    state["v"] = torch.zeros_like(p, dtype=torch.float32)
                
                m, v = state["m"], state["v"]
                t = state["step"] + 1
                # calculate ema
                m.mul_(beta_1).add_(g32, alpha=1-beta_1) # this updates m in-place and stores in self.state
                v.mul_(beta_2).addcmul_(g32, g32, value=1-beta_2)
                # calculate unbiased values
                m_hat = m / (1 - beta_1 ** t)
                v_hat = v / (1 - beta_2 ** t)
                # update weights
                denom = torch.sqrt(v_hat) + eps
                p.addcdiv_(m_hat, denom, value=-lr)
                # update step
                state["step"] += 1