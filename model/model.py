import numpy as np
import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from model.network.g_v2 import Gv2
from model.network.g_v2_three_class import Gv2ThreeClass
from model.network.g_v2_one_input import Gv2OneInput
import utils.utils as utils
import model.loss.loss as L
from data.dataset import XDataset
import model.metrics.metrics as M
import os
import json
from datetime import datetime as dt
from time import time
from torch.utils.tensorboard import SummaryWriter


class Model(object):
    def __init__(self, cfg):
        # ========= Config =========
        self.model_type = cfg.MODEL_TYPE
        if 'gv2' in self.model_type:
            self.g = Gv2(model_type=self.model_type)
        elif 'three-class' in self.model_type:
            self.g = Gv2ThreeClass(model_type=self.model_type)
        elif 'gv2-oi' in self.model_type:
            self.g = Gv2OneInput(model_type=self.model_type)
        self.addition_metrics_cd = cfg.COMMON.ADDITION_METRICS_CD

        # ========= Dataset =========
        dataset = XDataset(cfg.XRAY_PATH, cfg.GT_PATH, model_type=self.model_type, file_name=True, wm_type=cfg.TRAIN.WEIGHT_MAP_TYPE, wm_param=cfg.TRAIN.DWM_PARAMETER, edge_th=cfg.TRAIN.EDGE_THRESHOLD)
        num_total, num_one = len(dataset), len(dataset) // 5
        test_start_index_list = list(range(0, num_total, num_one))
        # 0 - 4
        cur_start_index = test_start_index_list[0]
        train_dataset, validate_dataset = dataset[0:cur_start_index] + dataset[cur_start_index + num_one:num_total], dataset[cur_start_index:cur_start_index + num_one]
        test_dataset = validate_dataset

        # ========= Phase Init =========
        if cfg.PHASE == 'train':
            # ====== Optimizer and Lr-Schedule ======
            self.g_optimizer = optim.Adam(
                self.g.parameters(),
                lr=cfg.TRAIN.G_LR,
                betas=cfg.TRAIN.G_BETAS
            )
            self.g_lr_scheduler = optim.lr_scheduler.MultiStepLR(
                self.g_optimizer,
                milestones=cfg.TRAIN.G_MILESTONES,
                gamma=cfg.TRAIN.G_GAMMA
            )
            # ====== Dataloader ======
            self.train_dataloader = data.DataLoader(
                dataset=train_dataset,
                batch_size=cfg.TRAIN.TRAIN_BATCH_SIZE,
                num_workers=4,
                pin_memory=True,
                shuffle=True,
                drop_last=True
            )
            self.validate_dataloader = data.DataLoader(
                dataset=validate_dataset,
                batch_size=cfg.TRAIN.VALIDATE_BATCH_SIZE,
                num_workers=4,
                pin_memory=True,
                shuffle=False
            )
            # ====== Output Path ======
            self.output_path = cfg.TRAIN.OUTPUT_PATH
            self.output_path = os.path.join(self.output_path, dt.now().strftime('%Y-%m-%d-%H:%M:%S') + '_' + '+'.join(self.model_type))
            utils.mkdir(self.output_path)
            # ====== Save Config ======
            with open(os.path.join(self.output_path, 'config.json'), mode='w') as f:
                json.dump(cfg, f)
            # ====== Config ======
            self.epoch_num = cfg.TRAIN.EPOCH
            self.checkpoint_save_frequency = cfg.TRAIN.CHECKPOINT_SAVE_FREQUENCY
            self.stop_signal = cfg.TRAIN.STOP_SIGNAL
            self.resume_weight_path = cfg.TRAIN.RESUME_WEIGHT_PATH
            self.metrics_choose = cfg.TRAIN.METRICS_CHOOSE
            sample_num = len(self.validate_dataloader)
            self.save_sample_list = np.unique(np.random.randint(0, sample_num, size=cfg.TRAIN.SAVE_RESULT_AMOUNT))
            self.phase = 'train'
        elif cfg.PHASE == 'test':
            # ====== Dataloader ======
            self.test_dataloader = data.DataLoader(
                dataset=test_dataset,
                batch_size=cfg.TEST.TEST_BATCH_SIZE,
                num_workers=4,
                pin_memory=True,
                shuffle=False
            )
            # ====== Output Path ======
            self.output_path = cfg.TEST.OUTPUT_PATH
            self.output_path = os.path.join(self.output_path, dt.now().strftime('%Y-%m-%d-%H:%M:%S') + '_' + '+'.join(self.model_type))
            # ====== Config ======
            self.weight_path = cfg.TEST.WEIGHT_PATH
            self.metrics_choose = cfg.TEST.METRICS_CHOOSE
            self.phase = 'test'
        else:
            raise Exception('[ERROR] 没有指定正确的阶段!(请指定成train或test)')

    def run(self):
        if self.phase == 'train':
            self._train()
        elif self.phase == 'test':
            self._test()

    def _train(self):
        # ========= Tensorboard and Checkpoint =========
        train_log_path = os.path.join(self.output_path, 'log', 'train')
        validate_log_path = os.path.join(self.output_path, 'log', 'validate')
        utils.mkdir(train_log_path)
        utils.mkdir(validate_log_path)
        train_writer = SummaryWriter(train_log_path)
        validate_writer = SummaryWriter(validate_log_path)
        checkpoint_path = os.path.join(self.output_path, 'checkpoint')
        utils.mkdir(checkpoint_path)

        self.g.apply(utils.init_weights)
        if cuda.is_available():
            self.g = nn.DataParallel(self.g).cuda()

        # ========= Index Init =========
        best_validate_metrics = -1
        best_epoch_index = -1
        pre_validate_metrics = -1
        start_epoch_index = 0

        # ========= Network Resume =========
        if self.resume_weight_path:
            checkpoint = torch.load(self.resume_weight_path)
            start_epoch_index = checkpoint['epoch_index'] + 1
            best_validate_metrics = checkpoint['best_metrics']
            best_epoch_index = checkpoint['best_epoch_index']
            self.g.load_state_dict(checkpoint['g_state_dict'])
            print(
                f'[INFO] Resume complete! Best {self.metrics_choose} is {best_validate_metrics} at epoch {best_epoch_index + 1}. Current epoch index is {start_epoch_index + 1}')

        # ========= Main Body =========
        for epoch_index in range(start_epoch_index, self.epoch_num):
            # ====== Run One Epoch ======
            validate_metrics = self._run_one_epoch(epoch_index=epoch_index, train_writer=train_writer,
                                                   validate_writer=validate_writer)
            # ====== Save Checkpoint ======
            if (epoch_index + 1) % self.checkpoint_save_frequency == 0 or abs(
                    validate_metrics - pre_validate_metrics) < self.stop_signal or validate_metrics > best_validate_metrics:
                checkpoint_save_path = os.path.join(checkpoint_path, f'checkpoint-{epoch_index + 1}.pth')
                if abs(validate_metrics - pre_validate_metrics) < self.stop_signal:
                    checkpoint_save_path = os.path.join(checkpoint_path, f'checkpoint-{epoch_index + 1}-last.pth')
                if validate_metrics > best_validate_metrics:
                    best_epoch_index = epoch_index
                    best_validate_metrics = validate_metrics
                    checkpoint_save_path = os.path.join(checkpoint_path, f'checkpoint-{epoch_index + 1}-best.pth')
                checkpoint = {
                    'epoch_index': epoch_index,
                    'best_metrics': best_validate_metrics,
                    'best_epoch_index': best_epoch_index,
                    'g_state_dict': self.g.state_dict()
                }
                torch.save(checkpoint, checkpoint_save_path)
                print(f'[INFO] saving checkpoint at [EPOCH {epoch_index + 1}] to [{checkpoint_save_path}]')
                if abs(validate_metrics - pre_validate_metrics) < self.stop_signal:
                    print(f'[INFO] train ended early')
                    break
            pre_validate_metrics = validate_metrics
        train_writer.close()
        validate_writer.close()

    def _test(self):
        test_log_path = os.path.join(self.output_path, 'log', 'test')
        utils.mkdir(test_log_path)
        test_writer = SummaryWriter(test_log_path)

        # ========= Network Init =========
        if cuda.is_available():
            self.g = nn.DataParallel(self.g).cuda()

        # ========= Network Load =========
        checkpoint = torch.load(self.weight_path)
        current_epoch_index = checkpoint['epoch_index']
        best_validate_metrics = checkpoint['best_metrics']
        best_epoch_index = checkpoint['best_epoch_index']
        self.g.load_state_dict(checkpoint['g_state_dict'])
        print(
            f'[INFO] Load complete! Best {self.metrics_choose} is {best_validate_metrics} at epoch {best_epoch_index + 1}. There are {current_epoch_index + 1} epoch in total')

        # ========= Run One Epoch =========
        self._run_one_epoch(test_writer=test_writer)
        test_writer.close()

    def _calculate_loss(self, xray, gt, wm=None, phase_type='train'):
        # ========= R Loss =========
        predict = self.g(xray)
        r_loss = L.RLoss(predict, gt, wm=wm)

        t_loss = r_loss

        product = dict()
        product['r_loss'] = r_loss
        product['t_loss'] = t_loss
        product['predict'] = predict
        return product

    def _update_statistics(self, statistics, ret_loss):
        statistics['r_statistics'].update(ret_loss['r_loss'].item())
        statistics['t_statistics'].update(ret_loss['t_loss'].item())

    def _update_metrics_statistics(self, predict, gt, m_statistics, cd_statistics=None, file_name=None):
        if torch.count_nonzero(predict) != 0 and torch.count_nonzero(gt) != 0:
            if self.metrics_choose == 'Dice':
                metrics = M.dice(predict, gt)
            elif self.metrics_choose == 'IoU':
                metrics = M.iou(predict, gt)
            else:
                raise Exception('[ERROR] wrong metric!')
            if cd_statistics is not None:
                cd_statistics.update(M.cd(predict, gt))
            m_statistics.update(metrics)
        else:
            print(f'[WARNING] maybe {file_name} has something wrong!')

    def _print_additional_loss(self, statistics, print_type, metrics_statistics=None, cd_statistics=None):
        if print_type == 'val':
            print(f'TLoss={statistics["t_statistics"].val:.3f}')
        elif print_type == 'avg':
            print(f'TLoss={statistics["t_statistics"].avg:.3f}')
        elif print_type == 'validate' and metrics_statistics is not None:
            print(f'{self.metrics_choose}={metrics_statistics.avg:.3f} ', end='')
            if cd_statistics is not None:
                print(f'CD={cd_statistics.avg:.3f} ', end='')
            print()
        elif print_type == 'test' and metrics_statistics is not None:
            print(f'{self.metrics_choose}={metrics_statistics.val:.3f} ', end='')
            if cd_statistics is not None:
                    print(f'CD={cd_statistics.val:.3f} ', end='')
            print()
        elif print_type == 'test_avg' and metrics_statistics is not None:
            print('[INFO] ', end='')
            print(f'{self.metrics_choose}={metrics_statistics.avg:.3f} ', end='')
            if cd_statistics is not None:
                print(f'CD={cd_statistics.avg:.3f} ', end='')
            print()

    def _write_summary(self, writer, statistics, index, print_type, metrics_statistics=None, cd_metrics_statistics=None, file_name=None):
        if print_type == 'val':
            writer.add_scalar('Batch/RLoss', statistics['r_statistics'].val, index)
        elif print_type == 'avg':
            writer.add_scalar('Epoch/RLoss', statistics['r_statistics'].avg, index)
        elif print_type == 'validate' and metrics_statistics is not None:
            writer.add_scalar('Epoch/RLoss', statistics['r_statistics'].avg, index)
            writer.add_scalar(f'Epoch/Metrics/{self.metrics_choose}', metrics_statistics.avg, index)
            if self.addition_metrics_cd and cd_metrics_statistics is not None:
                writer.add_scalar('Epoch/Metrics/CD', cd_metrics_statistics.avg, index)
        elif print_type == 'test' and metrics_statistics is not None:
            if file_name is not None:
                assert len(file_name) == 1
                file_name_one = file_name[0]
                file_name_index = int(file_name_one.split('_')[-1])
                writer.add_scalar('Sample/IndexOnSpine', file_name_index, index)
            writer.add_scalar('Sample/RLoss', statistics['r_statistics'].val, index)
            writer.add_scalar(f'Sample/Metrics/{self.metrics_choose}', metrics_statistics.val, index)
            if self.addition_metrics_cd and cd_metrics_statistics is not None:
                writer.add_scalar('Sample/Metrics/CD', cd_metrics_statistics.val, index)
        elif print_type == 'test_final' and metrics_statistics is not None:
            writer.add_scalar('Epoch/RLoss', statistics['r_statistics'].avg, index)
            writer.add_scalar(f'Epoch/Metrics/{self.metrics_choose}', metrics_statistics.avg, index)
            if self.addition_metrics_cd and cd_metrics_statistics is not None:
                writer.add_scalar('Epoch/Metrics/CD', cd_metrics_statistics.avg, index)

    def _save_result(self, save_path, file_name, predict, gt):
        predict_np = predict.detach().cpu().numpy()
        gt_np = gt.detach().cpu().numpy()

        save_sample_path = os.path.join(save_path, str(file_name))
        utils.mkdir(save_sample_path)
        save_predict_path = os.path.join(save_sample_path, 'predict')
        save_gt_path = os.path.join(save_sample_path, 'gt')
        np.savez_compressed(save_predict_path, predict=predict_np)
        np.savez_compressed(save_gt_path, gt=gt_np)

    def _run_one_epoch(self, epoch_index=None, train_writer=None, validate_writer=None, test_writer=None):
        # ========= One Epoch =========
        if self.phase == 'train':
            if epoch_index is None or train_writer is None or validate_writer is None:
                raise Exception('[ERROR] miss parameters!')

            # ====== PART Train ======
            self.g.train()

            # ====== Statistics Init ======
            epoch_time = utils.AverageMeter()
            batch_time = utils.AverageMeter()
            statistics = dict()
            statistics['r_statistics'] = utils.AverageMeter()
            statistics['t_statistics'] = utils.AverageMeter()

            batch_num = len(self.train_dataloader)
            train_epoch_start_time = time()

            # ====== One Batch (Train) ======
            for batch_index, (xray, gt, _, wm) in enumerate(self.train_dataloader):
                batch_start_time = time()
                # === Data to GPU ===
                with torch.autograd.set_detect_anomaly(True):

                    if cuda.is_available():
                        xray = xray.cuda(non_blocking=True)
                        gt = gt.cuda(non_blocking=True)
                        wm = wm.cuda(non_blocking=True)

                    # === Calculate Loss ===
                    product = self._calculate_loss(xray, gt, wm=wm, phase_type='train')

                    # === Backward ===
                    self.g.zero_grad()
                    product['t_loss'].backward()

                    self.g_optimizer.step()
                # === Update Statistics ===
                self._update_statistics(statistics, product)
                # === Tensorboard Record (Batch / Train) ===
                self._write_summary(train_writer, statistics, epoch_index * batch_num + batch_index + 1, print_type='val')
                # === Print Log (Batch / Train) ===
                batch_time.update(time() - batch_start_time)
                print(f'[INFO] [EPOCH {epoch_index + 1}/{self.epoch_num}] [Batch {batch_index + 1}/{batch_num}] '
                      f'BatchTime={batch_time.val:.3f}s RLoss={statistics["r_statistics"].val:.3f} ', end='')
                self._print_additional_loss(statistics, print_type='val')
            # ====== Lr Scheduler Step ======
            self.g_lr_scheduler.step()
            # ====== Tensorboard Record (Epoch / Train) ======
            self._write_summary(train_writer, statistics, epoch_index + 1, print_type='avg')
            # ====== Print Log (Epoch / Train) ======
            epoch_time.update(int(time() - train_epoch_start_time))
            print(f'[INFO] [EPOCH {epoch_index + 1}/{self.epoch_num}] EpochTime={epoch_time.val // 60}m{epoch_time.val % 60}s '
                  f'RLoss={statistics["r_statistics"].avg:.3f} ', end='')
            self._print_additional_loss(statistics, print_type='avg')

            # ====== PART Validate ======
            self.g.eval()

            # ====== Statistics Reset ======
            statistics['r_statistics'].reset()
            statistics['t_statistics'].reset()
            # ====== Metrics Init ======
            metrics_statistics = utils.AverageMeter()
            cd_metrics_statistics = utils.AverageMeter() if self.addition_metrics_cd else None

            sample_num = len(self.validate_dataloader)
            save_path = os.path.join(self.output_path, 'predict_and_gt', f'epoch-{epoch_index + 1}')

            # ====== One Sample (Validate) ======
            for sample_index, (xray, gt, file_name, wm) in enumerate(self.validate_dataloader):
                with torch.no_grad():
                    # === Data to GPU ===
                    if cuda.is_available():
                        xray = xray.cuda(non_blocking=True)
                        gt = gt.cuda(non_blocking=True)
                        wm = wm.cuda(non_blocking=True)
                    # === Calculate Loss and Get Product ===
                    product = self._calculate_loss(
                        xray, gt, wm=wm, phase_type='validate')
                    self._update_statistics(statistics, product)
                    # === Transfer Predict to Voxel-Model ===
                    predict = utils.predict_to_voxel(product['predict'])
                    # === Three Class ===
                    if 'three-class' in self.model_type:
                        gt[gt == 2] = 0
                    # === Calculate Metrics and Update Statistics ===
                    self._update_metrics_statistics(predict, gt, metrics_statistics, cd_statistics=cd_metrics_statistics, file_name=file_name)
                    # === Save Result ===
                    if sample_index in self.save_sample_list:
                        self._save_result(save_path, file_name, predict, gt)
            # ====== Tensorboard Record (Epoch / Validate) ======
            self._write_summary(validate_writer, statistics, epoch_index + 1, print_type='validate', metrics_statistics=metrics_statistics, cd_metrics_statistics=cd_metrics_statistics)
            # ====== Print Log (Epoch / Validate) ======
            print(f'[INFO] [EPOCH {epoch_index + 1}/{self.epoch_num}] [SAMPLE {sample_num}] '
                  f'RLoss={statistics["r_statistics"].avg:.3f} ', end='')
            self._print_additional_loss(statistics, print_type='validate', metrics_statistics=metrics_statistics, cd_statistics=cd_metrics_statistics)
            # ====== Return Metrics ======
            return metrics_statistics.avg

        elif self.phase == 'test':
            if test_writer is None:
                raise Exception('[ERROR] miss parameters!')

            # ====== PART Test ======
            self.g.eval()

            # ====== Statistics Init ======
            statistics = dict()
            statistics['r_statistics'] = utils.AverageMeter()
            statistics['t_statistics'] = utils.AverageMeter()
            metrics_statistics = utils.AverageMeter()
            cd_metrics_statistics = utils.AverageMeter() if self.addition_metrics_cd else None

            sample_num = len(self.test_dataloader)
            save_path = os.path.join(self.output_path, 'predict_and_gt')

            # ====== One Sample (Test) ======
            for sample_index, (xray, gt, file_name, wm) in enumerate(self.test_dataloader):
                with torch.no_grad():
                    # === Data to GPU ===
                    if cuda.is_available():
                        xray = xray.cuda(non_blocking=True)
                        gt = gt.cuda(non_blocking=True)
                        wm = wm.cuda(non_blocking=True)
                    # === Calculate Loss and Get Product ===
                    product = self._calculate_loss(
                        xray, gt, wm=wm, phase_type='test')
                    # === Update Statistics ===
                    self._update_statistics(statistics, product)
                    # === Transfer Predict to Voxel-Model ===
                    predict = utils.predict_to_voxel(product['predict'])
                    # === Three Class ===
                    if 'three-class' in self.model_type:
                        gt[gt == 2] = 0
                    # === Calculate Metrics and Update Statistics ===
                    self._update_metrics_statistics(predict, gt, metrics_statistics, cd_statistics=cd_metrics_statistics, file_name=file_name)
                    # === Save Result ===
                    self._save_result(save_path, file_name, predict, gt)
                    # ====== Tensorboard Record (Sample / Test) ======
                    self._write_summary(test_writer, statistics, sample_index + 1, print_type='test', metrics_statistics=metrics_statistics, cd_metrics_statistics=cd_metrics_statistics, file_name=file_name)
                    # ====== Print Log (Sample / Test) ======
                    print(f'[INFO] [SAMPLE {sample_index + 1}/{sample_num}] '
                          f'RLoss={statistics["r_statistics"].val:.3f} ', end='')
                    self._print_additional_loss(statistics, print_type='test',
                                                metrics_statistics=metrics_statistics, cd_statistics=cd_metrics_statistics)
            # ====== Print Log (Epoch / Test) ======
            self._print_additional_loss(statistics, print_type='test_avg',
                                        metrics_statistics=metrics_statistics, cd_statistics=cd_metrics_statistics)
