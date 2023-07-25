#!/bin/bash

echo "Running pre-processing script"

VIRTUAL_ENV=/tmp/nnunet_env 
source $VIRTUAL_ENV/bin/activate

tar -xf /tmp/c3d-nightly-Linux-x86_64_mini.tar.gz
PATH="/tmp/c3d-1.1.0-Linux-x86_64/bin/:$PATH"

mkdir -p /data/input/preprocessed

for file in $(ls /data/input/*_hires_mri.nii.gz);do

	echo $file
	filename_base_withdir=$(echo $file | rev | cut -d '_' -f 2- | rev) 

    filename_base=$(basename $filename_base_withdir)

    # Perform N4
    c3d $file -stretch 0.1% 99.9% 0 1000 -clip 0 1000 -o /tmp/${filename_base}_clip.nii.gz

    python /tmp/n4clip.py -i /tmp/${filename_base}_clip.nii.gz -o /tmp/${filename_base}_n4.nii.gz

    # Rescale the raw image to 0 - 1000 range (as input images already have)
    c3d /tmp/${filename_base}_n4.nii.gz -stretch 0.1% 99.9% 0 1000 -clip 0 1000 -o /data/input/preprocessed/${filename_base}_n4clip.nii.gz

    rm /tmp/${filename_base}_clip.nii.gz
    rm /tmp/${filename_base}_n4.nii.gz

done

