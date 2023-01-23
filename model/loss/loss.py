import torch
import torch.nn.functional as F


class WeightedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, predict, gt, wm=None):
        predict = predict.view(predict.size(0), predict.size(1), -1)
        predict = predict.transpose(1, 2)
        predict = predict.contiguous().view(-1, predict.size(2))
        gt = gt.view(-1, 1)

        predict = F.log_softmax(predict, dim=1)
        predict = predict.gather(1, gt)
        loss = -1 * predict
        if wm is not None:
            wm = wm.view(-1, 1)
            loss = loss * wm

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


def RLoss(predict, gt, weight=1., wm=None):
    return WeightedCrossEntropyLoss()(predict, gt, wm=wm) * weight
