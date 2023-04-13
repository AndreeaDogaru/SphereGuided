#!/bin/bash
dataset=$1
scene=$2
exp_name=$3

if [[ " $@ " =~ " sphere " ]]; then
  sphere_arg="--spheres_conf confs/sphere.yaml"
else
  sphere_arg=""
fi

if [[ "$dataset" == "nerf" ]]; then
    extra_arg="--no_filtering"
else
    extra_arg=""
fi

cd NeuS
python exp_runner.py --mode validate_mesh --is_continue --conf ./confs/womask_${dataset}.conf --case ${scene} --exp_name ${exp_name} ${sphere_arg} ${extra_arg}
cd ..