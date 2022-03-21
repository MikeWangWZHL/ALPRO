export PYTHONPATH="$PYTHONPATH:$PWD"
echo $PYTHONPATH

STEP='best'

CONFIG_PATH='config_release/msrvtt_ret.json'

TXT_DB='data/msrvtt_ret/txt/test.jsonl'
IMG_DB='data/msrvtt_ret/videos/TestVideo'

CUDA_VISIBLE_DEVICES=0,3

echo $TXT_DB

python load_pretrained_TimeSFormer.py \
      --do_inference 1 \
      --inference_split test \
      --inference_model_step $STEP \
      --inference_txt_db $TXT_DB \
      --inference_img_db $IMG_DB \
      --inference_batch_size 64 \
      --output_dir  output/downstreams/msrvtt_ret/public/ \
      --config $CONFIG_PATH