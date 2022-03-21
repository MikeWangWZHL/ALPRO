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

def video2txt(results, qid2data, qid_2_originalqid_mapping, original_video2txt_gt, top_k, vote_number):
    
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

    # get final prediction
    # using maximum aggregation: choose the top one confidence knowledge text 
    LOGGER.info("generating metric...")
    error_samples = []
    for vid, gt in original_video2txt_gt.items():
        if if_majority_vote:
            # use majority vote to choose prediction
            preds = video2txt_pred[vid][:vote_number*top_k]
            count_dict = defaultdict(int)
            for pred in preds:
                curr_pred_qid = pred[0]
                curr_pred_mapped_qid = qid_2_originalqid_mapping[str(curr_pred_qid)]
                count_dict[curr_pred_mapped_qid] += 1
            pred_list = [(mapped_qid, count) for mapped_qid,count in count_dict.items()]
            pred_list = sorted(pred_list, key = lambda x:x[1], reverse=True)

            final_predicts = [item[0] for item in pred_list[:top_k]]
            if gt['txt_id'] not in final_predicts:
                pred_ids_txts = [(item[0],qid2data[str(item[0])]['txt']) for item in preds]
                error_samples.append(
                    {
                        'vid_id':vid,
                        'preds':pred_ids_txts,
                        'gt':gt
                    }
                )
        else:
            candidates = set()
            idx = 0
            while len(candidates) != top_k:
                curr_pred_qid = video2txt_pred[vid][idx][0]
                curr_pred_mapped_qid = qid_2_originalqid_mapping[str(curr_pred_qid)]
                candidates.add(curr_pred_mapped_qid)
                idx += 1
            if gt['txt_id'] not in candidates:
                preds = video2txt_pred[vid][:idx]
                pred_ids_txts = [(item[0],qid2data[str(item[0])]['txt']) for item in preds]
                error_samples.append(
                    {
                        'vid_id':vid,
                        'preds':pred_ids_txts,
                        'gt':gt
                    }
                )
    print(f'recall top-{top_k}:',(len(original_video2txt_gt)-len(error_samples))/len(original_video2txt_gt))
    if if_majority_vote:
        with open(f'txt_augmented_video2txt_error_top-{top_k}_majority_vote_N-{vote_number}.json', 'w') as o:
            json.dump(error_samples, o, indent=4)
    else:
        with open(f'txt_augmented_video2txt_error_top-{top_k}.json', 'w') as o:
            json.dump(error_samples, o, indent=4)


def txt2video(results, originalqid_2_qid_mapping, original_txt2video_gt, top_k):
    # get raw prediction
    txt2video_pred = defaultdict(set) # omit duplicated qid result
    for item in results:
        vid = item['vid_id']
        qid = item['txt_id']
        score = item['score']
        sim = item['sim']
        txt2video_pred[qid].add((vid,score,sim))
    
    LOGGER.info("sorting...")
    for qid,preds in txt2video_pred.items():
        preds = list(preds)
        txt2video_pred[qid] = sorted(preds, key=lambda x:x[1], reverse=True)

    # get final prediction
    # using maximum aggregation: choose the top one confidence knowledge text 
    LOGGER.info("generating metric...")
    error_samples = []
    for original_qid, gt in original_txt2video_gt.items():
        qids = originalqid_2_qid_mapping[original_qid]
        if if_majority_vote:
            # use majority vote to choose prediction
            count_dict = defaultdict(int)
            for qid in qids:
                qid = int(qid)
                preds = txt2video_pred[qid][:top_k]
                for pred in preds:
                    count_dict[pred[0]] += 1
            prediction_list = [(vid, count) for vid,count in count_dict.items()]
            prediction_list = sorted(prediction_list, key=lambda x:x[1], reverse=True)
            final_predicts = [item[0] for item in prediction_list[:top_k]]
            if gt['vid_id'] not in final_predicts:
                error_samples.append(
                    {
                        'txt_id':original_qid,
                        'txt':gt['txt'],
                        'preds':final_predicts,
                        'gt':gt['vid_id']
                    }
                )
        else:
            raise NotImplementedError

    print(f'recall top-{top_k}:',(len(original_video2txt_gt)-len(error_samples))/len(original_video2txt_gt))
    if if_majority_vote:
        with open(f'txt_augmented_txt2video_error_top-{top_k}_majority_vote_N-{vote_number}.json', 'w') as o:
            json.dump(error_samples, o, indent=4)
    else:
        with open(f'txt_augmented_txt2video_error_top-{top_k}.json', 'w') as o:
            json.dump(error_samples, o, indent=4)
if __name__ == '__main__':

    original_ann_jsonl_path = "/shared/nas/data/m1/wangz3/video_language_pretraining_project/ALPRO/data/msrvtt_ret/txt/test.jsonl"
    result_json_path = "/shared/nas/data/m1/wangz3/video_language_pretraining_project/ALPRO/output/downstreams/msrvtt_ret/public/results_gpt-neo_augmented_test_reformat_2-7/step_best_1_mean/results.json"
    qid2data_json_path = "/shared/nas/data/m1/wangz3/video_language_pretraining_project/ALPRO/output/downstreams/msrvtt_ret/public/results_gpt-neo_augmented_test_reformat_2-7/step_best_1_mean/qid2data.json"

    top_k = 10
    if_majority_vote = True
    if if_majority_vote:
        LOGGER.info("using majority vote...")
    
    vote_number = 11
    LOGGER.info(f"top k = {top_k}")
    

    # get ground truth
    original_gt_datalist = mk_video_ret_datalist(load_jsonl(original_ann_jsonl_path))
    original_video2txt_gt = {item['vid_id']:{'txt_id':item['id'],'txt':item['txt']} for item in original_gt_datalist}
    original_txt2video_gt = {item['id']:{'vid_id':item['vid_id'],'txt':item['txt']} for item in original_gt_datalist}

    # map augmented qid to original qid:
    qid2data = json.load(open(qid2data_json_path))
    qid_2_originalqid_mapping = {}
    originalqid_2_qid_mapping = defaultdict(set)
    for qid,data in qid2data.items():
        vid_id = data['vid_id']
        original_qid = original_video2txt_gt[vid_id]['txt_id']
        qid_2_originalqid_mapping[qid] = original_qid
        originalqid_2_qid_mapping[original_qid].add(qid)
    
    # get raw results    
    LOGGER.info("loading results...")
    results = json.load(open(result_json_path))
    LOGGER.info(f"all results num: {len(results)}")

    ## video2txt
    # video2txt(results, qid2data, qid_2_originalqid_mapping, original_video2txt_gt, top_k, vote_number)
    
    # txt2video
    txt2video(results, originalqid_2_qid_mapping, original_txt2video_gt, top_k)
