#!/bin/bash

# generate desired camera path
python buildnet3d/scripts/generate_camera_path.py

# render the scene using the generated camera path
python buildnet3d/render/render.py \
    --load-config outputs/EPFL/semantic-sdf/2024-12-14_020842/config.yml \
    --load-camera-path outputs/cameras.json \
    --output-path outputs/rendered