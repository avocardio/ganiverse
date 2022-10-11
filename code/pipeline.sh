
# Complete run through

# !!! Manually start another tmux session to run lossbot

python code/preprocessing.py 'data/raw/*/*' 500
python code/mixup.py 'data/raw/*/*' 10 500
#python code/mixup.py 'data/augmented/*' 10 500
python code/gan_w.py 

# postprocessing....
