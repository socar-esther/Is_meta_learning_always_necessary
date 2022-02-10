### export path
export PYTHONPATH=$PYTHONPATH
nohup python3.9 -u -m train_script 1>logs/protonet.txt 2>&1 &