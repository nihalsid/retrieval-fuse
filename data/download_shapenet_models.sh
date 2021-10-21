#!/bin/bash

# get shapenet processed data
wget https://www.dropbox.com/s/5fnmi66gg26egmo/retrievalfuse_ShapeNetV2_trained.zip

# unzip
unzip retrievalfuse_ShapeNetV2_trained.zip

# copy
rsync -a retrievalfuse_ShapeNetV2_trained/retrieval data
mkdir -p runs/checkpoints
cp retrievalfuse_ShapeNetV2_trained/superres_retrieval_ShapeNetV2.ckpt runs/checkpoints/
cp retrievalfuse_ShapeNetV2_trained/surfacerecon_retrieval_ShapeNetV2.ckpt runs/checkpoints/
cp retrievalfuse_ShapeNetV2_trained/superres_refinement_ShapeNetV2.ckpt runs/checkpoints/
cp retrievalfuse_ShapeNetV2_trained/surfacerecon_refinement_ShapeNetV2.ckpt runs/checkpoints/

# remove
rm -rf retrievalfuse_ShapeNetV2_trained.zip
rm -rf retrievalfuse_ShapeNetV2_trained