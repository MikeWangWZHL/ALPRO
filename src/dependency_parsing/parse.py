import os
import json
from tqdm import tqdm
import spacy
import torch
from subject_verb_object_extract import findSVOs

def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]

def parse_caption(nlp, texts):
    if not isinstance(texts, list):
        texts = [texts]
    docs = nlp.pipe(texts)
    all_svo_lists = []
    all_ents = []
    for doc in docs:
        svo_list = findSVOs(doc)
        ents = [ent.text for ent in doc.ents]
        all_svo_lists.append(svo_list)
        all_ents.append(ents)
        # for token in doc:
        #     if token.pos_ == 'VERB'
        #     print()
            # print(token.text, token.dep_, token.head.text, token.head.pos_,
            #         [child for child in token.children])
        # print(svo_list,ents)
    return all_svo_lists, all_ents

def main():
    # ''' set up device '''
    # # use cuda
    # if torch.cuda.is_available():  
    #     dev = "cuda:3" 
    # else:  
    #     dev = "cpu"
    # # CUDA_VISIBLE_DEVICES=0,1,2,3
    # device = torch.device(dev)


    '''load original jsonl'''
    input_txt_jsonl = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/ALPRO/data/msrvtt_ret/txt/test.jsonl'
    output_txt_jsonl = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/ALPRO/src/dependency_parsing/parsed_data/msrvtt_parsed_test.jsonl'
    original_ann = load_jsonl(input_txt_jsonl)
    parsed_ann = []

    '''set up spacy pipeline'''
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("merge_entities")
    nlp.add_pipe("merge_noun_chunks")

    for item in tqdm(original_ann):
        
        query = item['caption']
        svo_list, ents = parse_caption(nlp, query)
        svo_list, ents = svo_list[0], ents[0]

        new_item = item.copy()
        new_item['subject_verb_object'] = svo_list
        new_item['named_entities'] = ents
        parsed_ann.append(new_item)


    '''output'''
    with open(output_txt_jsonl,'w') as out:
        for item in parsed_ann:
            out.write(json.dumps(item))
            out.write('\n')

if __name__ == '__main__':
    main()