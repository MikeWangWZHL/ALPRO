import json


vid2tags_json_path = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/ALPRO/src/knowledge_prompt/Azure_image_tagging_API/tag_results/text_frames_tags_02-13-22_22-15-50.json'
# vid2tags_json_path = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/ALPRO/src/knowledge_prompt/Azure_image_tagging_API/tag_results/text_frames_2_tags_02-14-22_12-10-53.json'
vid2tags = json.load(open(vid2tags_json_path))
top_k = 20

for vid, tags in vid2tags.items():
    print(vid)
    print()
    print(', '.join([tag[0] for tag in tags[:top_k]])) 
    print()
    print('###############')