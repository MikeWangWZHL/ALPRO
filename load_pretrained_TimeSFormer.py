import json
import math
import os
import random
import time
from collections import defaultdict
from os.path import exists, join

import horovod.torch as hvd
import numpy as np
import torch
import torch.nn.functional as F
from apex import amp
from easydict import EasyDict as edict
from src.configs.config import shared_configs
from src.datasets.data_utils import ImageNorm, mk_input_group
from src.datasets.dataloader import InfiniteIterator, PrefetchLoader
from src.datasets.dataset_video_retrieval import (
    AlproVideoRetrievalDataset, AlproVideoRetrievalEvalDataset,
    VideoRetrievalCollator)
from src.modeling.alpro_models import AlproForVideoTextRetrieval
from src.optimization.sched import get_lr_sched
from src.optimization.utils import setup_e2e_optimizer
from src.utils.basic_utils import (get_rounded_percentage, load_json,
                                   load_jsonl, save_json)
from src.utils.distributed import all_gather_list
from src.utils.load_save import E2E_TrainingRestorer as TrainingRestorer
from src.utils.load_save import (ModelSaver,
                                 load_state_dict_with_pos_embed_resizing,
                                 save_training_meta)
from src.utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from src.utils.misc import NoOp, set_random_seed, zero_none_grad
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import BertConfig, BertTokenizerFast
from src.modeling.timesformer.vit import TimeSformer 

cfg = shared_configs.get_video_retrieval_args()
device = None
print(cfg)

LOGGER.info("Setup model...")
# has to be a BertConfig instance
model_cfg = load_json(cfg.model_config)
model_cfg = BertConfig(**model_cfg)

# add downstream model config
add_attr_list = []
for k in add_attr_list:
    setattr(model_cfg, k, cfg[k])

# we separate the CNN and the transformer in order to use different optimizer for each
# transformer still has a CNN layer inside, used to down sample grid.
LOGGER.info("setup e2e model")

video_enc_cfg = load_json(cfg.visual_model_cfg)

video_enc_cfg['num_frm'] = cfg.num_frm
video_enc_cfg['img_size'] = cfg.crop_img_size

visual_encoder = TimeSformer(model_cfg=video_enc_cfg, input_format='RGB', cross_attention_config=model_cfg)

print(visual_encoder)

# model = AlproForVideoTextRetrieval(
#     model_cfg, 
#     input_format=cfg.img_input_format,
#     video_enc_cfg=video_enc_cfg
#     )
# if cfg.e2e_weights_path:
#     LOGGER.info(f"Loading e2e weights from {cfg.e2e_weights_path}")
#     num_patches = (cfg.crop_img_size // video_enc_cfg['patch_size']) ** 2
#     # NOTE strict if False if loaded from ALBEF ckpt
#     load_state_dict_with_pos_embed_resizing(model, 
#                                             cfg.e2e_weights_path, 
#                                             num_patches=num_patches, 
#                                             num_frames=cfg.num_frm, 
#                                             strict=False,
#                                             )
# else:
#     LOGGER.info(f"Loading visual weights from {cfg.visual_weights_path}")
#     LOGGER.info(f"Loading bert weights from {cfg.bert_weights_path}")
#     model.load_separate_ckpt(
#         visual_weights_path=cfg.visual_weights_path,
#         bert_weights_path=cfg.bert_weights_path
#     )

# # if cfg.freeze_cnn:
# #     model.freeze_cnn_backbone()
# model.to(device)

# LOGGER.info("Setup model done!")

# print()
# print()
# print(model.visual_encoder)