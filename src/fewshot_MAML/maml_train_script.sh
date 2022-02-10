### export path
export PYTHONPATH=$PYTHONPATH
nohup python3.9 -u -m maml_train 1>logs/maml.txt 2>&1 &