import torch
import torch.nn as nn
from model.network.se import SEch


class Gv2(nn.Module):
    def __init__(self, model_type=None):
        super(Gv2, self).__init__()
        self.model_type = model_type if model_type is not None else list()

        def _make_layer(in_ch, out_ch, block_type, compress_size=None, scale_factor=None):
            if block_type == 'base1':
                return nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True)
                )
            elif block_type == 'base2':
                return nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm3d(out_ch)
                )
            elif block_type == 'down':
                return nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True)
                )
            elif block_type == 'up':
                return nn.Sequential(
                    nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True)
                )
            elif block_type == 'compress':
                if compress_size is None:
                    raise Exception('[ERROR] compress size should be specified!')
                else:
                    return nn.Sequential(
                        nn.AdaptiveMaxPool3d(compress_size),
                        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                        nn.BatchNorm3d(out_ch),
                        nn.ReLU(inplace=True)
                    )
            elif block_type == 'expand':
                if scale_factor is None:
                    raise Exception('[ERROR] scale factor should be specified!')
                else:
                    return nn.Sequential(
                        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                        nn.BatchNorm3d(out_ch),
                        nn.ReLU(inplace=True),
                        nn.Upsample(scale_factor=scale_factor, mode='nearest')
                    )
            elif block_type == 'skip':
                return nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True)
                )
            elif block_type == 'aggregate':
                return nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True)
                )
            elif block_type == 'se':
                return nn.Sequential(
                    SEch(in_ch)
                )
            elif block_type == 'guide-conv':
                return nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool3d(1)
                )
            elif block_type == 'guide-fc':
                return nn.Sequential(
                    nn.Linear(in_ch, in_ch, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Linear(in_ch, in_ch, bias=False),
                    nn.Sigmoid()
                )
            else:
                raise Exception('[ERROR] not implemented this type of block!')

        # ================== Baseline ==================
        self.base1 = _make_layer(2, 8, 'base1')
        self.down1 = _make_layer(8, 16, 'down')
        self.down2 = _make_layer(16, 32, 'down')
        self.down3 = _make_layer(32, 64, 'down')
        self.down4 = _make_layer(64, 128, 'down')
        self.up1 = _make_layer(128, 64, 'up')
        self.up2 = _make_layer(64, 32, 'up')
        self.up3 = _make_layer(32, 16, 'up')
        self.up4 = _make_layer(16, 8, 'up')
        self.base2 = _make_layer(8, 2, 'base2')


        # ================== with Fullscale Shortcut ==================
        if 'fullscale&guide-v2-shortcut' in self.model_type:
            self.compress1_8 = _make_layer(8, 64 // 4, 'compress', compress_size=16)
            self.compress1_16 = _make_layer(16, 64 // 4, 'compress', compress_size=16)
            self.compress1_32 = _make_layer(32, 64 // 4, 'compress', compress_size=16)
            self.skip1_64 = _make_layer(64, 64 // 4, 'skip')
            self.aggregate1 = _make_layer(128, 64, 'aggregate')

            self.compress2_8 = _make_layer(8, 32 // 4, 'compress', compress_size=32)
            self.compress2_16 = _make_layer(16, 32 // 4, 'compress', compress_size=32)
            self.skip2_32 = _make_layer(32, 32 // 4, 'skip')
            self.expand2_64 = _make_layer(64, 32 // 4, 'expand', scale_factor=2)
            self.aggregate2 = _make_layer(64, 32, 'aggregate')

            self.compress3_8 = _make_layer(8, 16 // 4, 'compress', compress_size=64)
            self.skip3_16 = _make_layer(16, 16 // 4, 'skip')
            self.expand3_32 = _make_layer(32, 16 // 4, 'expand', scale_factor=2)
            self.expand3_64 = _make_layer(64, 16 // 4, 'expand', scale_factor=4)
            self.aggregate3 = _make_layer(32, 16, 'aggregate')

            self.skip4_8 = _make_layer(8, 8 // 4, 'skip')
            self.expand4_16 = _make_layer(16, 8 // 4, 'expand', scale_factor=2)
            self.expand4_32 = _make_layer(32, 8 // 4, 'expand', scale_factor=4)
            self.expand4_64 = _make_layer(64, 8 // 4, 'expand', scale_factor=8)
            self.aggregate4 = _make_layer(16, 8, 'aggregate')

            self.guide1_conv = _make_layer(128, 64, 'guide-conv')
            self.guide2_conv = _make_layer(64, 32, 'guide-conv')
            self.guide3_conv = _make_layer(32, 16, 'guide-conv')
            self.guide4_conv = _make_layer(16, 8, 'guide-conv')

            self.guide1_fc = _make_layer(64, 64, 'guide-fc')
            self.guide2_fc = _make_layer(32, 32, 'guide-fc')
            self.guide3_fc = _make_layer(16, 16, 'guide-fc')
            self.guide4_fc = _make_layer(8, 8, 'guide-fc')

    def forward(self, x):
        # ================== Encoder ==================
        x_left1 = self.base1(x)
        x_left2 = self.down1(x_left1)
        x_left3 = self.down2(x_left2)
        x_left4 = self.down3(x_left3)

        # ================== Middle Stage ==================
        x_middle = self.down4(x_left4)

        # ================== Decoder ==================
        if 'no-shortcut' in self.model_type:
            x_right1 = self.up1(x_middle)
            x_right2 = self.up2(x_right1)
            x_right3 = self.up3(x_right2)
            x_right4 = self.up4(x_right3)
            y = self.base2(x_right4)
            return y
        elif 'fullscale&guide-v2-shortcut' in self.model_type:
            x_1_8 = self.compress1_8(x_left1)
            x_1_16 = self.compress1_16(x_left2)
            x_1_32 = self.compress1_32(x_left3)
            x_1_64 = self.skip1_64(x_left4)
            x_1_up = self.up1(x_middle)
            b, c, _, _, _ = x_1_up.size()
            x_1_cat = torch.cat((x_1_8, x_1_16, x_1_32, x_1_64, x_1_up), 1)
            x_1_guide_conv = self.guide1_conv(x_1_cat).reshape(b, c)
            x_1_guide_fc = self.guide1_fc(x_1_guide_conv).reshape(b, c, 1, 1, 1)
            x_1_supple = torch.cat((x_1_8, x_1_16, x_1_32, x_1_64), 1) * x_1_guide_fc
            x_right1 = self.aggregate1(
                torch.cat((x_1_supple, x_1_up), 1))

            x_2_8 = self.compress2_8(x_left1)
            x_2_16 = self.compress2_16(x_left2)
            x_2_32 = self.skip2_32(x_left3)
            x_2_64 = self.expand2_64(x_left4)
            x_2_up = self.up2(x_right1)
            b, c, _, _, _ = x_2_up.size()
            x_2_cat = torch.cat((x_2_8, x_2_16, x_2_32, x_2_64, x_2_up), 1)
            x_2_guide_conv = self.guide2_conv(x_2_cat).reshape(b, c)
            x_2_guide_fc = self.guide2_fc(x_2_guide_conv).reshape(b, c, 1, 1, 1)
            x_2_supple = torch.cat((x_2_8, x_2_16, x_2_32, x_2_64), 1) * x_2_guide_fc
            x_right2 = self.aggregate2(
                torch.cat((x_2_supple, x_2_up), 1))

            x_3_8 = self.compress3_8(x_left1)
            x_3_16 = self.skip3_16(x_left2)
            x_3_32 = self.expand3_32(x_left3)
            x_3_64 = self.expand3_64(x_left4)
            x_3_up = self.up3(x_right2)
            b, c, _, _, _ = x_3_up.size()
            x_3_cat = torch.cat((x_3_8, x_3_16, x_3_32, x_3_64, x_3_up), 1)
            x_3_guide_conv = self.guide3_conv(x_3_cat).reshape(b, c)
            x_3_guide_fc = self.guide3_fc(x_3_guide_conv).reshape(b, c, 1, 1, 1)
            x_3_supple = torch.cat((x_3_8, x_3_16, x_3_32, x_3_64), 1) * x_3_guide_fc
            x_right3 = self.aggregate3(
                torch.cat((x_3_supple, x_3_up), 1))

            x_4_8 = self.skip4_8(x_left1)
            x_4_16 = self.expand4_16(x_left2)
            x_4_32 = self.expand4_32(x_left3)
            x_4_64 = self.expand4_64(x_left4)
            x_4_up = self.up4(x_right3)
            b, c, _, _, _ = x_4_up.size()
            x_4_cat = torch.cat((x_4_8, x_4_16, x_4_32, x_4_64, x_4_up), 1)
            x_4_guide_conv = self.guide4_conv(x_4_cat).reshape(b, c)
            x_4_guide_fc = self.guide4_fc(x_4_guide_conv).reshape(b, c, 1, 1, 1)
            x_4_supple = torch.cat((x_4_8, x_4_16, x_4_32, x_4_64), 1) * x_4_guide_fc
            x_right4 = self.aggregate4(
                torch.cat((x_4_supple, x_4_up), 1))
            y = self.base2(x_right4)
            return y
        else:
            raise Exception('[ERROR] wrong network keyword!')
