import os
import openai
import json
from tqdm import tqdm
import time

def get_response(config):
    response = openai.Completion.create(
        engine=config['engine'], 
        prompt=config['prompt'], 
        max_tokens=config['max_tokens'],
        temperature=config['temperature'],
        top_p=config['top_p'],
        n=config['n'],
        stream=config['stream'],
        logprobs=config['logprobs'],
        stop=config['stop']
    )
    return response

def load_knowledge_prompt_template(path):
    ret = ''
    with open(path) as f:
        for line in f:
            ret += line
    return ret

def construct_prompt(prefix, input_query, suffix = 'Knowledge:'):
    input_query = input_query.strip()
    if not input_query.endswith('.'):
        input_query += '.'
    ret = prefix + input_query + '\n' + suffix
    return ret

def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]

def main():
    # Load your API key from an environment variable or secret management service
    # openai.api_key = os.getenv("OPENAI_API_KEY")
    '''gpt3 api config'''
    openai.api_key = "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"
    config = {
        "engine": "text-davinci-001",
        "prompt": None,
        "max_tokens": 64,
        "temperature": 0.7,
        "top_p": 1,
        "n": 10,
        "best_of": 3,
        "stream": False,
        "logprobs": None,
        "stop": None
    }
    '''load original jsonl'''
    input_txt_jsonl = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/ALPRO/data/msrvtt_ret/txt/test.jsonl'
    output_txt_jsonl = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/ALPRO/src/knowledge_prompt/augmented_data/msrvtt_ret/augmented_test.jsonl'
    original_ann = load_jsonl(input_txt_jsonl)
    augmented_ann = []
    
    '''query api'''
    prompt_path = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/ALPRO/src/knowledge_prompt/prompts/p_test2.txt'
    prompt_template = load_knowledge_prompt_template(prompt_path)
    for item in tqdm(original_ann):
        query = item['caption']
        config['prompt'] = construct_prompt(prompt_template, query, suffix = 'Knowledge:')
        # print(config['prompt'])
        response = get_response(config)
        knowledge_texts = [s['text'].strip() for s in response['choices']]
        # if not query.endswith('.'):
        #     query += '.'
        # new_caption = query + ' ' + knowledge_text
        new_item = item.copy()
        new_item['knowledge'] = knowledge_texts
        augmented_ann.append(new_item)
        time.sleep(1.2)
        break

    '''output'''
    with open(output_txt_jsonl,'w') as out:
        for item in augmented_ann:
            out.write(json.dumps(item))
            out.write('\n')


if __name__ == '__main__':
    main()