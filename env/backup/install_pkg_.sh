apt update
apt install lsof

# conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

pip install -U pip setuptools wheel
pip install -U 'spacy[cuda112]'
python -m spacy download en_core_web_sm


conda install -c conda-forge cudatoolkit-dev -y
HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_NCCL_LINK=SHARED HOROVOD_WITH_PYTORCH=1 HOROVOD_GPU_OPERATIONS=NCCL
pip install --no-cache-dir horovod -y

pip uninstall pillow -y && \
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

pip install -r requirements.txt

git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./

