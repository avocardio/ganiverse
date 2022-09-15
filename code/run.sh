
# Complete run through

# !!! Manually start another tmux session to run lossbot

python code/preprocessing.py data/raw/nasa/
python code/mixup.py data/raw/nasa/ 2000
python code/mixup.py data/augmented/ 2000
python code/gan_w.py 

# postprocessing....
