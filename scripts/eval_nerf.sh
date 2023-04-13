#!/bin/bash
method=$1
scene=$2
exp_name=$3

case $method in
    neus)
        path=NeuS/exps/${exp_name}/${scene}/womask_nerf/meshes/00300000.ply
        ;;
    nwarp)
        path=NeuralWarp/evals/${exp_name}/NeuralWarp_nerf_${scene}/output_mesh.ply
        ;;
    *)
        echo "Nerf synthetic evaluation is not implemented for $method"
        exit 1
        ;;
esac

python eval_nerf.py --data_dir ./data/nerf --mesh_path ${path} --scene ${scene}


