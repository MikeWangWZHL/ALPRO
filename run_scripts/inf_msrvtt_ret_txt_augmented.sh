cd ..

export PYTHONPATH="$PYTHONPATH:$PWD"
echo $PYTHONPATH

STEP='best'

CONFIG_PATH='config_release/msrvtt_ret_txt_augmented.json'

TXT_DB='src/knowledge_prompt/augmented_data/msrvtt_ret/gpt-neo_augmented_test_reformat.jsonl'
IMG_DB='data/msrvtt_ret/videos/TestVideo'

export CUDA_VISIBLE_DEVICES="0,2,3"

horovodrun -np 3 python src/tasks/run_video_retrieval.py \
      --do_inference 1 \
      --inference_split test \
      --inference_model_step $STEP \
      --inference_txt_db $TXT_DB \
      --inference_img_db $IMG_DB \
      --inference_batch_size 64 \
      --output_dir  output/downstreams/msrvtt_ret/public \
      --config $CONFIG_PATH