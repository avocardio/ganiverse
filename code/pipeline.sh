
# Complete run through

# !!! Manually start another tmux session to run lossbot

python code/preprocessing.py
python code/mixup.py data/raw/*/* 2000
python code/mixup.py data/augmented/* 2000
python code/gan_w.py 

# postprocessing....
