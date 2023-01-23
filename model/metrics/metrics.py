import torch
import sklearn.metrics
import numpy as np
from chamfer_distance import ChamferDistance
import mcubes


def iou(predict, gt):
    intersection = torch.sum(predict.mul(gt)).type(torch.float32)
    union = torch.sum(torch.ge(predict.add(gt), 1)).type(torch.float32)

    return (intersection / union).item()


def dice(predict, gt):
    intersection = torch.sum(predict.mul(gt)).type(torch.float32)
    union = torch.sum(torch.ge(predict.add(gt), 1)).type(torch.float32)

    return ((2 * intersection) / (union + intersection)).item()


def ap(predict, gt):
    predict_clone = predict.clone().cpu().numpy()
    gt_clone = gt.clone().cpu().numpy()

    batch_size = predict_clone.shape[0]
    precisions = []
    for i in range(batch_size):
        predict_one = predict_clone[i, ...].flatten()
        gt_one = gt_clone[i, ...].flatten()
        precisions.append(sklearn.metrics.average_precision_score(gt_one, predict_one))
    avg_precision = np.array(precisions).mean()
    return avg_precision.item()


def cd(predict, gt):
    predict_clone = predict.clone().cpu().numpy()
    gt_clone = gt.clone().cpu().numpy()

    batch_size = predict_clone.shape[0]
    cd_metrics = 0.
    for i in range(batch_size):
        gt_vertices, _ = mcubes.marching_cubes(gt_clone[i].squeeze(), 0)
        predict_vertices, _ = mcubes.marching_cubes(predict_clone[i].squeeze(), 0)
        gt_pc = torch.tensor(gt_vertices).type(torch.float32).unsqueeze(dim=0).cuda()
        predict_pc = torch.tensor(predict_vertices).type(torch.float32).unsqueeze(dim=0).cuda()

        cal_cd = ChamferDistance()
        dist1, dist2, idx1, idx2 = cal_cd(gt_pc, predict_pc)
        cd_metrics += torch.mean(dist1) + torch.mean(dist2)
    return (cd_metrics / batch_size).item()
