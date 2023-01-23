import os
from PIL import Image
import torch
import torch.utils.data.dataset as dataset
import torch.nn.functional as F
import numpy as np
import glob
import utils.utils as utils
from scipy import ndimage


class XDataset(dataset.Dataset):
    def __init__(self, xray_path, gt_path, model_type=None, file_name=False, wm_type=0, wm_param=None, edge_th=2.):
        super(XDataset, self).__init__()
        xray_path_f = os.path.join(xray_path, 'front')
        xray_path_s = os.path.join(xray_path, 'side')
        self.xray_f = sorted(glob.glob(os.path.join(xray_path_f, '*.png')), key=lambda x: x[:-4])
        self.xray_s = sorted(glob.glob(os.path.join(xray_path_s, '*.png')), key=lambda x: x[:-4])
        self.gt_files = sorted(glob.glob(os.path.join(gt_path, '*.npz')), key=lambda x: x[:-4])
        print(f'[INFO] dataset contains {len(self.gt_files)} samples')
        self.file_name = file_name
        if file_name:
            self.file_name_list = [x.split(os.sep)[-1][:-4] for x in self.gt_files]
        self.model_type = model_type if model_type is not None else list()
        self.wm_type = wm_type
        self.wm_param = wm_param if wm_param is not None else [3., 1., 9.]
        self.edge_th = edge_th

    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, idx):
        # ============ Gt ============
        gt_np = np.load(self.gt_files[idx])['gt']
        gt = torch.from_numpy(gt_np).type(torch.int64)
        # ====== Pad ======
        # (96, 128, 128) to (128) * 3
        correct_shape = (128, 128, 128)
        if gt.shape != correct_shape:
            front_ = (correct_shape[0] - gt.shape[0]) // 2
            behind_ = correct_shape[0] - gt.shape[0] - front_
            up_ = (correct_shape[1] - gt.shape[1]) // 2
            blow_ = correct_shape[1] - gt.shape[1] - up_
            left_ = (correct_shape[2] - gt.shape[2]) // 2
            right_ = correct_shape[2] - gt.shape[2] - left_
            gt = F.pad(gt, (left_, right_, up_, blow_, front_, behind_), mode='constant', value=0)
        # ====== Three Class ======
        if 'three-class' in self.model_type:
            mask = gt.numpy().astype(np.int64)
            distance = ndimage.distance_transform_edt(mask == 0)
            distance[distance > self.edge_th] = 0
            distance[distance != 0] = 1
            edge = (distance * 2).astype(np.int64)
            gt = torch.from_numpy(mask + edge).type(torch.int64)
        # ====== Wm ======
        if self.wm_type == 0:
            wm = torch.ones_like(gt)
        else:
            # weight map
            wm = utils.distance_weight_map(gt.numpy(), threshold=self.wm_param[0], alpha=self.wm_param[1], gamma=self.wm_param[2])
            wm = torch.from_numpy(wm).type(torch.float32)

        # ============ Xray ============
        front_np = np.array(Image.open(self.xray_f[idx]), dtype=np.float32)
        side_np = np.array(Image.open(self.xray_s[idx]), dtype=np.float32)
        # ====== Z-score ======
        front_np = (front_np - front_np.mean()) / front_np.std()
        side_np = (side_np - side_np.mean()) / side_np.std()
        front = torch.from_numpy(front_np).type(torch.float32).unsqueeze(0)
        side = torch.from_numpy(side_np).type(torch.float32).unsqueeze(0)
        # ============ Align ============
        expand_size = 128
        front = front.repeat(expand_size, 1, 1).unsqueeze(dim=0)
        if 'align' in self.model_type:
            side = side.squeeze(dim=0).unsqueeze(dim=2).repeat(1, 1, expand_size).unsqueeze(dim=0)
        else:
            side = side.repeat(expand_size, 1, 1).unsqueeze(dim=0)
        # ============ One Input ============
        if 'one-input' in self.model_type:
            if 'front' in self.model_type:
                xray = front
            elif 'side' in self.model_type:
                xray = side
            else:
                raise Exception('[ERROR] please specify front or side (one-input model)')
        else:
            xray = torch.cat((front, side), dim=0)

        if self.file_name:
            file_name = self.file_name_list[idx]
            return xray, gt, file_name, wm
        else:
            return xray, gt, None, wm
