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

def get_predictions(results, mode = 'video2txt'):
    if mode == 'video2txt':
        # get raw prediction
        video2txt_pred = defaultdict(set) # omit duplicated qid result
        for item in results:
            vid = item['vid_id']
            qid = item['txt_id']
            score = item['score']
            sim = item['sim']
            video2txt_pred[vid].add((qid,score,sim))
        
        LOGGER.info("sorting...")
        for vid,preds in video2txt_pred.items():
            preds = list(preds)
            video2txt_pred[vid] = sorted(preds, key=lambda x:x[1], reverse=True)

        return video2txt_pred



if __name__ == '__main__':

    for top_k in [1,5,10]:

        original_ann_jsonl_path = "/shared/nas/data/m1/wangz3/video_language_pretraining_project/ALPRO/data/msrvtt_ret/txt/test.jsonl"
        augmented_result_json_path = "/shared/nas/data/m1/wangz3/video_language_pretraining_project/ALPRO/output/downstreams/msrvtt_ret/public/results_gpt-neo_augmented_test_reformat_2-7/step_best_1_mean/results.json"

        qid2data_json_path = "/shared/nas/data/m1/wangz3/video_language_pretraining_project/ALPRO/output/downstreams/msrvtt_ret/public/results_gpt-neo_augmented_test_reformat_2-7/step_best_1_mean/qid2data.json"
        original_result_json_path = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/ALPRO/output/downstreams/msrvtt_ret/public/results_test/step_best_1_mean_original/results.json'

        
        show_k = top_k*10
        
        # get ground truth
        original_gt_datalist = mk_video_ret_datalist(load_jsonl(original_ann_jsonl_path))
        original_video2txt_gt = {item['vid_id']:{'txt_id':item['id'],'txt':item['txt']} for item in original_gt_datalist}
        original_txt2video_gt = {item['id']:{'vid_id':item['vid_id'],'txt':item['txt']} for item in original_gt_datalist}

        qid2data = json.load(open(qid2data_json_path))

        # get raw results    
        LOGGER.info("loading results...")
        original_results = json.load(open(original_result_json_path))
        augmented_results = json.load(open(augmented_result_json_path))
        LOGGER.info(f"all original results num: {len(original_results)}")
        LOGGER.info(f"all augmented results num: {len(augmented_results)}")

        # get prediction
        original_predictions = get_predictions(original_results, mode = 'video2txt')
        augmented_predictions = get_predictions(augmented_results, mode = 'video2txt')

        # load error 
        original_error_path = f'/shared/nas/data/m1/wangz3/video_language_pretraining_project/ALPRO/src/error_analysis/video2txt_error_top-{top_k}.json'
        augmented_error_path = f'/shared/nas/data/m1/wangz3/video_language_pretraining_project/ALPRO/src/knowledge_prompt/txt_augmented_video2txt_error_top-{top_k}_majority_vote_N-11.json'
        output_path = f'/shared/nas/data/m1/wangz3/video_language_pretraining_project/ALPRO/src/error_analysis/error_comparison/error_comparison-{top_k}.json'
        
        # compare error
        get_better = [] 
        get_worse = []

        original_error = json.load(open(original_error_path))
        augmented_error = json.load(open(augmented_error_path))

        original_error_vids = [item['vid_id'] for item in original_error]
        augmented_error_vids = [item['vid_id'] for item in augmented_error]

        for vid in original_error_vids:
            if vid not in augmented_error_vids:
                get_better.append(vid)
        for vid in augmented_error_vids:
            if vid not in original_error_vids:
                get_worse.append(vid)

        error_comparison = {
            'get_better':[{
                'vid_id':vid_id,
                'original_pred':[original_txt2video_gt[item[0]]['txt'] for item in original_predictions[vid_id][:top_k]],
                'augmented_pred':[qid2data[str(item[0])]['txt'] for item in augmented_predictions[vid_id][:show_k]],
                'gt':original_video2txt_gt[vid_id]
            } for vid_id in get_better],
            'get_worse':[{
                'vid_id':vid_id,
                'original_pred':[original_txt2video_gt[item[0]]['txt'] for item in original_predictions[vid_id][:top_k]],
                'augmented_pred':[qid2data[str(item[0])]['txt'] for item in augmented_predictions[vid_id][:show_k]],
                'gt':original_video2txt_gt[vid_id]
            } for vid_id in get_worse],

        }
        with open(output_path, 'w') as out:
            json.dump(error_comparison, out, indent=4)

        print(get_better)
        print()
        print(get_worse)