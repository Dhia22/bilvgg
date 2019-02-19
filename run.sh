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
python3 -B main.py -t --batch_size 40 --epochs 200 --base_learning_rate=0.01 --decay_factor=1 --decay_epochs=100 --no_gpus=1
