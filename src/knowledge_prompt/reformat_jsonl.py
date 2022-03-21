import json

# reformat jsonl: dplit each line contains multiple knowledge text,
# into several lines

input_jsonl = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/ALPRO/src/knowledge_prompt/augmented_data/msrvtt_ret/gpt-neo_augmented_test.jsonl'
output_jsonl = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/ALPRO/src/knowledge_prompt/augmented_data/msrvtt_ret/gpt-neo_augmented_test_reformat.jsonl'

data = []
new_data = []
with open(input_jsonl) as f:
    for line in f:
        data.append(json.loads(line))

for item in data:
    old_caption = item['caption'] 
    if not old_caption.endswith('.'):
        old_caption += '.'
    
    new_data.append({
        "caption":old_caption,
        "clip_name":item['clip_name'],
        "retrieval_key": item["retrieval_key"]
    }) # add the caption without knowledge as well
    
    for i in range(len(item['knowledge'])):
        kg_txt = item['knowledge'][i]
        new_data.append({
            "caption":old_caption + ' ' + kg_txt,
            "clip_name":item['clip_name'],
            "retrieval_key": item["retrieval_key"] + f"_{i}"
        })

with open(output_jsonl,'w') as out:
    for item in new_data:
        out.write(json.dumps(item))
        out.write('\n')