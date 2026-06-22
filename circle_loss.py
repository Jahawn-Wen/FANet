#from https://github.com/TinyZeaMays/CircleLoss/blob/master/circle_loss.py

from typing import Tuple

import torch
from torch import nn, Tensor


def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]



class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss

# TODO Focal loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
        # NOTE:
        # `sp` / `sn` here are cosine similarities from `convert_label_to_similarity()`.
        # For normalized features, similarities are in [-1, 1]. The previous implementation
        # used `log(sp)` directly, which becomes NaN when `sp <= 0`. We map similarities
        # into a probability-like space (0, 1) first to keep training stable.
        eps = 1e-6
        sp_p = ((sp + 1.0) * 0.5).clamp(eps, 1.0 - eps)  # positives should be close to 1
        sn_p = ((sn + 1.0) * 0.5).clamp(eps, 1.0 - eps)  # negatives should be close to 0

        # Positive pairs: encourage sp_p -> 1
        p_loss = -self.alpha * (1.0 - sp_p).pow(self.gamma) * torch.log(sp_p)
        # Negative pairs: encourage sn_p -> 0
        n_loss = -(1.0 - self.alpha) * (sn_p).pow(self.gamma) * torch.log(1.0 - sn_p)

        # Some batches may have no positive pairs (all labels unique) or no negative pairs.
        # Avoid mean() over empty tensors -> NaN.
        losses = []
        if p_loss.numel():
            losses.append(p_loss)
        if n_loss.numel():
            losses.append(n_loss)
        if not losses:
            # Keep dtype/device and avoid breaking autograd (returns a constant 0).
            return sp_p.sum() * 0.0
        return torch.cat(losses).mean()
    

class WeightNet(nn.Module):
    def __init__(self, input_size, hidden_size=100):
        super(WeightNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()  # Output weights in the [0, 1] range.
        )

    def forward(self, x):
        return self.fc(x)




if __name__ == "__main__":
    feat = nn.functional.normalize(torch.rand(256, 64, requires_grad=True))
    lbl = torch.randint(high=10, size=(256,))

    inp_sp, inp_sn = convert_label_to_similarity(feat, lbl)

    criterion = CircleLoss(m=0.25, gamma=256)
    
    circle_loss = criterion(inp_sp, inp_sn)

    print(circle_loss)
