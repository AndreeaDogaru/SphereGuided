#!/bin/bash
dataset=$1
scene=$2
scene_name=$(printf "scan%04d" $scene)
exp_name=$3

if [[ "$dataset" != "dtu" ]]; then
    echo "Error: Unisurf is only implemented for the DTU dataset"
    exit 1
fi

if [[ " $@ " =~ " sphere " ]]; then
  sphere_arg="--spheres-conf configs/sphere.yaml"
else
  sphere_arg=""
fi

cd unisurf
python train.py configs/DTU/${scene_name}.yaml --exp-name ${exp_name} ${sphere_arg}
cd ..
