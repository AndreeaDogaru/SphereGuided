#!/bin/bash
dataset=$1
scene=$2
scene_name=$(printf "scan%04d" $scene)
exp_name=$3
iteration=450000

if [[ " $@ " =~ " sphere " ]]; then
  sphere_arg="--spheres-conf confs/sphere.yaml"
else
  sphere_arg=""
fi

cd unisurf
python extract_mesh.py configs/DTU/${scene_name}.yaml --eval-iter $iteration --exp-name ${exp_name} --filter post
cd ..
