import os
from decord import VideoReader
import json
from glob import glob
import random
from PIL import Image
import numpy as np
import random
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPVisionModel, CLIPTokenizer, CLIPTextModel, CLIPModel, CLIPFeatureExtractor
import torch
from sklearn.cluster import KMeans

# TODO use CLIP image embedding, cluster video frames using K-means, sample one frame from each cluster
def CLIP_selection(video_path, clip_model, clip_processor, num_frm, sampling_frm_rate = 15):  
    # print('vp:',video_path)
    vr = VideoReader(video_path)
    vlen = len(vr)
    # calculate vision embedding for each frame
    all_frames = vr.get_batch(range(vlen)).asnumpy()
    all_frames_pil = [Image.fromarray(frm) for frm in all_frames]
    inputs = clip_processor(images=all_frames_pil, return_tensors="pt")
    outputs = clip_model(**inputs)
    pooled_output = outputs.pooler_output  # pooled CLS states
    frm_embeddings = pooled_output.detach().numpy()
    # cluster into num_frm clusters
    kmeans = KMeans(n_clusters=num_frm, random_state=0).fit(frm_embeddings)
    labels = kmeans.labels_ 
    frame_indices = []
    
    # random sample one frame from each cluster
    for i in range(num_frm):
        masked = np.where(labels == i)[0]
        frame_indices.append(np.random.choice(masked))
    raw_sample_frms = [all_frames_pil[idx] for idx in frame_indices]
    return frame_indices, raw_sample_frms


def random_selection(video_path, num_frm):
    vr = VideoReader(video_path)
    vlen = len(vr)
    frame_indices = sorted(random.sample(range(vlen), num_frm))
    raw_sample_frms = vr.get_batch(frame_indices).asnumpy()
    return frame_indices, raw_sample_frms

def save_frame(raw_sample_frms, frame_indices, video_name, output_dir):
    frames_dir = os.path.join(output_dir, video_name)
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    for i in range(len(raw_sample_frms)):
        frm = raw_sample_frms[i]
        idx = frame_indices[i]
        output_path = os.path.join(frames_dir, f'{video_name}_frame-{idx}.jpg')
        if isinstance(frm, Image.Image):
            im = frm
        elif isinstance(frm, np.ndarray):
            im = Image.fromarray(frm)
        else:
            raise ValueError(
                "frm must of type `PIL.Image.Image` or `np.ndarray`"
            )
        im.save(output_path)

if __name__ == "__main__":
    
    video_dir = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/ALPRO/data/msrvtt_ret/videos/TestVideo'
    method = "clip" 
    # method = "random"
    num_frm = 8
    video_paths = sorted(glob(os.path.join(video_dir,'*')))
    output_dir = f'./selected_frames_{method}'

    if method == 'clip':
        model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        for vp in tqdm(video_paths):
            frame_indices, raw_sample_frms = CLIP_selection(vp, model, processor, num_frm)
            # save frames
            video_name = os.path.basename(vp)
            save_frame(raw_sample_frms, frame_indices, video_name, output_dir)

    elif method == 'random':
        for vp in tqdm(video_paths):
            frame_indices, raw_sample_frms = random_selection(vp, num_frm)
            # save frames
            video_name = os.path.basename(vp)
            save_frame(raw_sample_frms, frame_indices, video_name, output_dir)
