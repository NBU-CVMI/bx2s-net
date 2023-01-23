from config import cfg
import os

# ========= GPU-ID Init =========
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.DEVICE

import utils.utils as utils
from model.model import Model
from time import time


# ================== Program Entry ==================
if __name__ == '__main__':
    # ========= Args Init =========
    args = utils.parse_arguments()

    if args.weight_path:
        cfg.PHASE = 'test'
        cfg.TEST.WEIGHT_PATH = args.weight_path
    else:
        cfg.PHASE = 'train'
    if args.resume_path:
        cfg.TRAIN.RESUME_WEIGHT_PATH = args.resume_path

    # ========= Fix Random Seed =========
    utils.fix_seed()

    # ========= Main Runtime =========
    print(f'{"-"*25}{cfg.PHASE.upper()}{"-"*25}')
    start_time = time()

    model = Model(cfg)
    model.run()

    intervals = int(time() - start_time)
    print(f'{"-"*25}{intervals // 3600}h {intervals % 3600 // 60}m {intervals % 60}s{"-"*25}')
