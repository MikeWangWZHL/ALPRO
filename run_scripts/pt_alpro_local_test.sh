# cd ..

# export PYTHONPATH="$PYTHONPATH:$PWD"
# echo $PYTHONPATH

# CONFIG_PATH='config_release/pretrain_alpro.json'

# horovodrun -np 16 python src/pretrain/run_pretrain_sparse.py \
#       --config $CONFIG_PATH \
#       --output_dir /export/home/workspace/experiments/alpro/vl/$(date '+%Y%m%d%H%M%S')


cd ..

export PYTHONPATH="$PYTHONPATH:$PWD"
echo $PYTHONPATH

CONFIG_PATH='config_release/pretrain_alpro_local_test_webvidonly.json'

export CUDA_VISIBLE_DEVICES=0,3

horovodrun -np 2 python src/pretrain/run_pretrain_sparse.py \
      --config $CONFIG_PATH \
      --output_dir output/experiments/pretrain_alpro_local_test_webvidonly