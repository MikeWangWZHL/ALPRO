import os
import json
from basic_utils import (get_rounded_percentage, load_json,
                                   load_jsonl, save_json)
from logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file

def mk_video_ret_datalist(raw_datalist):
    """
    Args:
        raw_datalist: list(dict)
        Each data point is {id: int, txt: str, vid_id: str}

    Returns:

    """
    LOGGER.info(f"Loaded data size {len(raw_datalist)}")

    datalist = []
    qid = 0
    for raw_d in raw_datalist:
        d = dict(
            id=qid,
            txt=raw_d["caption"],
            vid_id=raw_d["clip_name"]
        )
        qid += 1
        datalist.append(d)
    LOGGER.info(f"datalist {len(datalist)}")
    return datalist

if __name__ == '__main__':
    result_dir = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/ALPRO/output/downstreams/msrvtt_ret/public/results_test/step_best_1_mean'
    anno_jsonl = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/ALPRO/data/msrvtt_ret/txt/test.jsonl'
    result_json = os.path.join(result_dir,'results.json')
    
    # get ground truth
    gt_datalist = mk_video_ret_datalist(load_jsonl(anno_jsonl))
    video2txt_gt = {item['vid_id']:{'txt_id':item['id'],'txt':item['txt']} for item in gt_datalist}
    txt2video_gt = {item['id']:{'vid_id':item['vid_id'],'txt':item['txt']} for item in gt_datalist}
    
    # get prediction
    video2txt_top1_pred = {}
    results = json.load(open(result_json))
    for item in results:
        vid = item['vid_id']
        tid = item['txt_id']
        score = item['score']
        sim = item['sim']
        if vid not in video2txt_top1_pred:
            video2txt_top1_pred[vid] = {'txt_id':tid, 'score':score, 'sim':sim}
        else:
            if score > video2txt_top1_pred[vid]['score']:
                video2txt_top1_pred[vid] = {'txt_id':tid, 'score':score, 'sim':sim}
    
    # get error items
    # check vid2txt recall
    error_samples = []
    for key,pred in video2txt_top1_pred.items():
        pred_txt_id = pred['txt_id']
        gt_txt_id = video2txt_gt[key]['txt_id']
        if pred_txt_id != gt_txt_id:
            error_samples.append(
                {
                    'vid_id':key, 
                    'pred_id':pred_txt_id, 
                    'pred_txt':txt2video_gt[pred_txt_id]['txt'],
                    'gt_id':gt_txt_id,
                    'gt_txt':video2txt_gt[key]['txt']
                }
            )
    print(error_samples)
    print('num of gt:', len(gt_datalist))
    print('num of pred:', len(video2txt_top1_pred))
    print('recall 1:',(len(video2txt_top1_pred)-len(error_samples))/len(video2txt_top1_pred))

    with open('video2txt_error.json', 'w') as o:
        json.dump(error_samples, o, indent=4)
        