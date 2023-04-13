#!/bin/bash
dataset=$1
scene=$2
exp_name=$3

if [[ " $@ " =~ " sphere " ]]; then
  sphere_arg_base="--spheres_conf confs/sphere.yaml"
  sphere_arg_nw="--spheres_conf confs/sphere_nw.yaml"
else
  sphere_arg_base=""
  sphere_arg_nw=""
fi

cd NeuralWarp
echo "Training baseline VolSDF"
python train.py --conf confs/baseline_${dataset}.conf --scene ${scene} --exp_name ${exp_name} ${sphere_arg_base}
echo "Finetune using NeuralWarp"
python train.py --conf confs/NeuralWarp_${dataset}.conf --scene ${scene} --exp_name ${exp_name} ${sphere_arg_nw}
cd ..