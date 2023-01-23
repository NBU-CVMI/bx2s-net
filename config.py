from easydict import EasyDict

__C = EasyDict()
cfg = __C


# ================== Model Type ==================
# # BX2S
__C.MODEL_TYPE = ['baseline', 'gv2', 'no-shortcut']  # baseline + bx2s
# __C.MODEL_TYPE = ['baseline', 'align', 'gv2', 'no-shortcut']  # baseline + bx2s + align
# __C.MODEL_TYPE = [baseline', 'gv2', 'fullscale&guide-v2-shortcut']  # baseline + bx2s + ffag
# __C.MODEL_TYPE = ['baseline', 'align', 'gv2', 'fullscale&guide-v2-shortcut']  # baseline + bx2s + align + ffag
# __C.MODEL_TYPE = ['baseline', 'align', 'three-class', 'fullscale&guide-v2-shortcut', 'wm1810']  # baseline + bx2s + align + ffag + wm1810
# __C.MODEL_TYPE = ['baseline', 'gv2-oi', 'fullscale&guide-v2-shortcut', 'one-input', 'front']  # baseline + bx2s + align + ffag + one-input + front
# __C.MODEL_TYPE = ['baseline', 'gv2-oi', 'fullscale&guide-v2-shortcut', 'one-input', 'side']  # baseline + bx2s + align + ffag + one-input + side

# ================== Phase ==================
__C.PHASE = ''

# ================== Gpu-ID and Input-File-Path ==================
__C.DEVICE = '0'
# should contain front and side subfolders
__C.XRAY_PATH = ''
__C.GT_PATH = ''

# ================== Common ==================
__C.COMMON = EasyDict()

__C.COMMON.ADDITION_METRICS_CD = True

# ================== Train ==================
__C.TRAIN = EasyDict()

__C.TRAIN.OUTPUT_PATH = ''

__C.TRAIN.EPOCH = 50
__C.TRAIN.TRAIN_BATCH_SIZE = 4
__C.TRAIN.VALIDATE_BATCH_SIZE = 1
__C.TRAIN.CHECKPOINT_SAVE_FREQUENCY = 1
__C.TRAIN.SAVE_RESULT_AMOUNT = 10
__C.TRAIN.STOP_SIGNAL = float('-inf')
__C.TRAIN.METRICS_CHOOSE = 'Dice'
__C.TRAIN.RESUME_WEIGHT_PATH = ''
# ========= Adam Optimizer =========
__C.TRAIN.G_LR = 1e-2
__C.TRAIN.G_BETAS = (.9, .999)
# ========= Lr Schedule =========
__C.TRAIN.G_MILESTONES = [10, 20, 30, 40]
__C.TRAIN.G_GAMMA = .1
# ========= Weight Map Type =========
__C.TRAIN.WEIGHT_MAP_TYPE = 0
for i in __C.MODEL_TYPE:
    if 'wm' in i:
        __C.TRAIN.WEIGHT_MAP_TYPE = 1
__C.TRAIN.DWM_PARAMETER = [1.8, 1., 0.]
# ========= Edge Threshold =========
__C.TRAIN.EDGE_THRESHOLD = 1.8

# ================== Test ==================
__C.TEST = EasyDict()

__C.TEST.OUTPUT_PATH = ''

__C.TEST.TEST_BATCH_SIZE = 1
__C.TEST.METRICS_CHOOSE = 'Dice'
__C.TEST.WEIGHT_PATH = ''
