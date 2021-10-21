#!/bin/bash

# get shapenet processed data
wget https://www.dropbox.com/s/z93hlkmvsuzdw5c/retrievalfuse_ShapeNetV2_data.zip

# unzip
unzip retrievalfuse_ShapeNetV2_data.zip

# copy data
cp retrievalfuse_ShapeNetV2_data/occupancy/* data/occupancy/
cp retrievalfuse_ShapeNetV2_data/size/* data/size/
rsync -a retrievalfuse_ShapeNetV2_data/pc_20K/ShapeNetV2/ data/pc_20K/ShapeNetV2/
rsync -a retrievalfuse_ShapeNetV2_data/sdf_008/ data/sdf_008/ShapeNetV2/
rsync -a retrievalfuse_ShapeNetV2_data/sdf_064/ data/sdf_064/ShapeNetV2/

rm -rf retrievalfuse_ShapeNetV2_data.zip
rm -rf retrievalfuse_ShapeNetV2_data