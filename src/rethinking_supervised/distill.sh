### export path
export PYTHONPATH=$PYTHONPATH
nohup python3.9 -u -m train_distillation_script 1>logs/distill.txt 2>&1 &