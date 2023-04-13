#!/bin/bash
method=$1
scene=$2
exp_name=$3

case $method in
    neus)
        path=NeuS/exps/${exp_name}/dtu_scan${scene}/womask_dtu/meshes/00300000.ply
        ;;
    nwarp)
        path=NeuralWarp/evals/${exp_name}/NeuralWarp_dtu_${scene}/output_mesh.ply
        ;;
    unisurf)
        path=unisurf/${exp_name}/DTU/$(printf "scan%04d" $scene)/extraction/450000/meshes_postfiltered/scan_world_scale.ply
        ;;
    *)
        echo "Unknown method: $method"
        exit 1
        ;;
esac

python eval_dtu.py --data_dir ./data/dtu_eval --mesh_path ${path} --scene ${scene}


