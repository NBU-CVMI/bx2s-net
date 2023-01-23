import numpy as np
import os
import torch
import torch.backends.cudnn
from argparse import ArgumentParser
import random
from scipy import ndimage


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def confidence_to_voxel(vertebra_np):
    return vertebra_np[..., 1].copy()


def predict_to_voxel(predict):
    if predict.shape[1] == 3:
        predict_3 = torch.argmax(predict, 1)
        predict_3[predict_3 == 2] = 0
        return predict_3.type(torch.int64)
    else:
        return torch.argmax(predict, 1).type(torch.int64)


def init_weights(m):
    if type(m) == torch.nn.Conv2d or type(m) == torch.nn.Conv3d or \
       type(m) == torch.nn.ConvTranspose2d or type(m) == torch.nn.ConvTranspose3d:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.BatchNorm3d:
        torch.nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        # 注释原因：与SE模块中的bias=False冲突 !FIXED!
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def parse_arguments():
    parser = ArgumentParser(description='Parser of blank')
    parser.add_argument('--xray-path',
                        dest='xray_path',
                        type=str)
    parser.add_argument('--gt-path',
                        dest='gt_path',
                        type=str)
    parser.add_argument('--train-output-path',
                        dest='train_output_path',
                        type=str)
    parser.add_argument('--batch-size',
                        dest='batch_size',
                        type=int)
    parser.add_argument('--epoch',
                        dest='epoch',
                        type=int)
    parser.add_argument('--cp-save-freq',
                        dest='cp_save_freq',
                        type=int)
    parser.add_argument('--re-save-freq',
                        dest='re_save_freq',
                        type=int)
    parser.add_argument('--stop-signal',
                        dest='stop_signal',
                        type=float)
    parser.add_argument('--train-metrics-choose',
                        dest='train_metrics_choose',
                        type=str)
    parser.add_argument('--resume-path',
                        dest='resume_path',
                        type=str)
    parser.add_argument('--test-output-path',
                        dest='test_output_path',
                        type=str)
    parser.add_argument('--test-metrics-choose',
                        dest='test_metrics_choose',
                        type=str)
    parser.add_argument('--weight-path',
                        dest='weight_path',
                        type=str)
    args = parser.parse_args()
    return args


def fix_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def distance_weight_map(mask, threshold=2., alpha=1., gamma=10.):
    weight = ndimage.distance_transform_edt(mask == 0)
    weight[weight > threshold] = threshold
    wm = alpha * (1. - weight / threshold) + 1.
    wm[wm == 2.] = 1.
    edge = wm

    return edge
