import os
import json
from basic_utils import (get_rounded_percentage, load_json,
                                   load_jsonl, save_json)
from logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from collections import defaultdict

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
    result_dir = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/ALPRO/output/downstreams/msrvtt_ret/public/results_test/step_best_1_mean_original'
    anno_jsonl = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/ALPRO/data/msrvtt_ret/txt/test.jsonl'
    result_json = os.path.join(result_dir,'results.json')
    top_k = 10

    # get ground truth
    gt_datalist = mk_video_ret_datalist(load_jsonl(anno_jsonl))
    video2txt_gt = {item['vid_id']:{'txt_id':item['id'],'txt':item['txt']} for item in gt_datalist}
    txt2video_gt = {item['id']:{'vid_id':item['vid_id'],'txt':item['txt']} for item in gt_datalist}
    
    # get prediction
    video2txt_pred = defaultdict(list)
    results = json.load(open(result_json))
    for item in results:
        vid = item['vid_id']
        tid = item['txt_id']
        score = item['score']
        sim = item['sim']
        video2txt_pred[vid].append((tid,score,sim))
    
    for vid,preds in video2txt_pred.items():
        video2txt_pred[vid] = sorted(preds, key=lambda x:x[1], reverse=True)

    # get error items
    # check vid2txt recall
    error_samples = []
    for key, gt in video2txt_gt.items():
        preds = video2txt_pred[key][:top_k]
        hit = False
        for pred in preds:
            if pred[0] == gt['txt_id']:
                hit = True
                break
        if not hit:
            pred_ids_txts = [(item[0],txt2video_gt[item[0]]['txt']) for item in preds]
            
            error_samples.append(
                {
                    'vid_id':key,
                    'preds':pred_ids_txts,
                    'gt':gt
                }
            )
    print(error_samples)
    print('num of gt:', len(gt_datalist))
    print('num of pred:', len(video2txt_pred))
    print(f'recall top-{top_k}:',(len(video2txt_pred)-len(error_samples))/len(video2txt_pred))

    with open(f'video2txt_error_top-{top_k}.json', 'w') as o:
        json.dump(error_samples, o, indent=4)

