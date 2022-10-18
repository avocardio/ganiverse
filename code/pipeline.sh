
# Complete run through

# !!! Manually start another tmux session to run lossbot

python code/preprocessing.py 'data/raw/*/*' 256
python code/mixup.py 'data/raw/*/*' 5000 256
#python code/mixup.py 'data/augmented/*' 10 500
python code/gan_w_256_4m.py

python code/gan_w_256_1.8m.py

# postprocessing....
