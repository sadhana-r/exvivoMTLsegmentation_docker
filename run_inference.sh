#!/bin/bash

echo "Setting up the model for prediction..."

VIRTUAL_ENV=/tmp/nnunet_env 
source $VIRTUAL_ENV/bin/activate


##Set up standard nnUnet paths
export nnUNet_raw_data_base="/tmp/nnUnet_required/nnUnet_raw_data_base"
export nnUNet_preprocessed="/tmp/nnUnet_required/nnUNet_preprocessed"
export RESULTS_FOLDER="/tmp/nnUnet_required/nnUNet_trained_models"

#Need to add an empty channel to each of the input images since
# The modified nnUnet with Laplacian is written for a 2 channel input
echo "Preparing input images for inference..."

if [[ ! -d /tmp/c3d-1.1.0-Linux-x86_64 ]]; then
	tar -xf c3d-nightly-Linux-x86_64_mini.tar.gz
fi

PATH="c3d-1.1.0-Linux-x86_64/bin/:$PATH"

for file in $(ls /data/input/*_0000.nii.gz);do
	echo $file
	filename_base=$(echo $file | rev | cut -d '_' -f 2- | rev) 
	echo "Writing to " ${filename_base}_0001.nii.gz
	c3d $file -thresh 0 inf 0 0 -o ${filename_base}_0001.nii.gz
done

#Run nnUnet inference
echo "Generating MTL segmentation predictions..."
nnUNet_predict -i /data/input \
    -o /data/output \
    -t 601 -m 3d_fullres -tr nnUNetTrainerV2_SOR_MTLAtlas --disable_mixed_precision -f all

# rm /data/input/*_0001.nii.gz

echo "Finished predictions"
