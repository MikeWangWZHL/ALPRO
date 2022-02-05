import os
from decord import VideoReader
import json
from glob import glob
import random
from PIL import Image
import numpy as np
import random
from tqdm import tqdm

video_dir = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/ALPRO/data/msrvtt_ret/videos/TestVideo'
output_dir = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/ALPRO/src/error_analysis/msrvtt_ret_test_frames'
num_frm = 4
video_paths = sorted(glob(os.path.join(video_dir,'*')))

for vp in tqdm(video_paths):    
    video_name = os.path.basename(vp)
    frames_dir = os.path.join(output_dir, video_name)
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    vr = VideoReader(vp)
    vlen = len(vr)
    frame_indices = sorted(random.sample(range(vlen), num_frm))
    raw_sample_frms = vr.get_batch(frame_indices).asnumpy()
    for i in range(len(raw_sample_frms)):
        frm = raw_sample_frms[i]
        idx = frame_indices[i]
        output_path = os.path.join(frames_dir, f'{video_name}_frame-{idx}.jpg')
        im = Image.fromarray(frm)
        im.save(output_path)
