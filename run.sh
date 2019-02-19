#!bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda/cuda/lib64:/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
python3 -B main.py -t --batch_size 40 --epochs 200 --base_learning_rate=0.0001 --decay_factor=0.1 --decay_epochs=100 --no_gpus=1 --setup_meta

python3 -B main.py -e --batch_size 70 --part=fusion

python3 -B main.py -t --batch_size 70 --epochs 200 --base_learning_rate=0.001 --decay_factor=0.1 --decay_epochs=100 --no_gpus=1 --part=fusion


!git clone https://github.com/Dhia22/bilvgg.git
%cd bilvgg
!pip3 install scikit-learn==0.18
!wget http://41.229.96.242/uploads/vgg16_weights.npz
!wget http://41.229.96.242/uploads/datag.tar.gz
!tar -v -xvzf datag.tar.gz
!python3 -B main.py -t --batch_size 40 --epochs 200 --base_learning_rate=0.01 --decay_factor=1 --decay_epochs=100 --no_gpus=1


# memory footprint support libraries/code
!ln -sf /opt/bin/nvidia-smi /usr/bin/nvidia-smi
!pip install gputil
!pip install psutil
!pip install humanize
import psutil
import humanize
import os
import GPUtil as GPU
GPUs = GPU.getGPUs()
# XXX: only one GPU on Colab and isn’t guaranteed
gpu = GPUs[0]
def printm():
 process = psutil.Process(os.getpid())
 print("Gen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), " | Proc size: " + humanize.naturalsize( process.memory_info().rss))
 print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))
printm()
