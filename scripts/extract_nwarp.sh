#!/bin/bash
dataset=$1
scene=$2
exp_name=$3

if [[ "$@" =~ "sphere" ]]; then
  sphere_arg="--spheres_conf confs/sphere_nw.yaml"
else
  sphere_arg=""
fi

if [[ "$dataset" == "nerf" ]]; then
    extra_arg="--no_masks --no_refine_bb --no_one_cc"
else
    extra_arg=""
fi

cd NeuralWarp
python extract_mesh.py --conf confs/NeuralWarp_${dataset}.conf --scene ${scene} --exp_name ${exp_name} ${sphere_arg} ${extra_arg}
cd ..