import torch.nn as nn
import torch

class SMLoss(nn.Module):

    def __init__(self):
        super(SMLoss,self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self,densepose,targets):

        loss = torch.sum(torch.cat([self.l1_loss(densepose[i], targets[i]).unsqueeze(0) for i in range(24)]))

        return loss