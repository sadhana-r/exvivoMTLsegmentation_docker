# MTL Segmentation in ex vivo MRI
Docker for Laplace constrained MTL segmentation in PICSL 9.4T ex vivo MRI scans

# About

This is the set of files and scripts required to generate the docker container for predicting medial temporal lobe (MTL) segmentations on 9.4T ex vivo MRI scans of the temporal lobe. The network is based on a modified implementation of the nn-UNet framework which incorporated a Laplacian solver during end-to-end training to improve sulci detetction. If using this tool to segment the MTL, please consider citing the following paper:

``
Ravikumar, S., Ittyerah, R., Lim, S., Xie, L., Das, S., Khandelwal, P., ... & Yushkevich, P. A. (2023, June). Improved Segmentation of Deep Sulci in Cortical Gray Matter Using a Deep Learning Framework Incorporating Laplace’s Equation. In International Conference on Information Processing in Medical Imaging (pp. 692-704). Cham: Springer Nature Switzerland.
``

The trained model is saved on the cluster along with the c3d binaries required by the inference/pre-processing script. The official container on DockerHub is labeled `sadhanar/9.4exvivomtlseg:v1.0`. `v1.0` only segments the MTL region within the scan. If interested in segmenting the entire scan, pull `sadhanaravikumar/9.4exvivomtlseg:full` instead. (N.B. The `full` version was trained on older data which did not include the full extent of the SRLM)

The predicted multi-label segmentation consists of four labels: Gray Matter (1 - red), White Matter (2 - green) and Background (3 - blue) and SRLM (4 - yellow). 

<p float="left">
  <img align = "top" src="/img/exampleseg_mtl.png" width="45%" />
  <img aign = "top" src="/img/exampleseg_full.png" width="45%" /> 
</p>

# How to generate segmentations

## Step 1: Download the docker image

     docker pull sadhanaravikumar/9.4exvivomtlseg:v1.0 

or if interested in segmenting the entire scan, use

    docker pull sadhanaravikumar/9.4exvivomtlseg:full

## Step 2: Pre-process and prepare the data (Optional)

Before running inference, the data can be pre-processed and axis-aligned to faciliate manual editing.  

1. Run the pre-processing script (`run_preprocessing.sh`) which applies N4 bias correction and intensity normalization. This step is optional.
   
         docker run --user $(id -u) -it -v /path/to/inputdata/:/data/input sadhanaravikumar/9.4exvivomtlseg:v1.0 /bin/bash run_preprocessing.sh

2. Axis-align the scans

   a. Download the reference template image for axis-aigning the scans from [here](https://upenn.box.com/s/f4h0p96543dd3mx00bayag0u4s5iamap)
   
   b. Load the template image as the main image and MRI scan as an additional image. Enter manual registration mode and re-orient and align the scan to match the template image. More details on this process are provided [here](https://upenn.box.com/s/49co5uog6jl587tptan54pqnqdif9r0r)
   
   c. Save the transformation file and **make sure to save the resliced image with a filename ending in _0000.nii.gz**

## Step 3: Generate segmentation predictions

To generate MTL segmentation predictions, first log on to the lambda machines. Then run the following command which executes the `run_inference.sh` script on your data. Modify `/path/to/inputdata` to your directory where the input data ending in `*_0000.nii.gz` is located, and modify `/path/to/outputdata` to where you want to save the output predictions. The input and output data is located in `/data` inside the docker container. Leave the rest of the command as is. 

    docker run --user $(id -u) -it --gpus all --privileged -v /path/to/inputdata/:/data/input -v /path/to/outputdata/:/data/output sadhanaravikumar/9.4exvivomtlseg:v1.0 /bin/bash run_inference.sh 

# Sample Data

A sample dataset is located [here](https://upenn.box.com/s/zlj5r2pcvuqct5ynwf4znak3k3ky9jz3)

# Running with Apptainer/Singularity
In environments where Docker is not available you may be able to run the pipeline using Apptainer (aka singularity). These instructions have been tested with Apptainer version 1.1.9.

Use this command to pull the container from Docker hub and build a sandbox container. Sandbox mode is necessary because the container must be writable.

    singularity build --sandbox /somepath/9.4exvivomtlseg_v1.0_sandbox docker://sadhanaravikumar/9.4exvivomtlseg:v1.0

Then you can run the inference script as follows:

    singularity exec -B /path/to/inputdata/:/data/input:rw -B /path/to/outputdata/:/data/output:rw \
        --no-mount tmp --pwd /tmp --writable-tmpfs --nv \
        /somepath/9.4exvivomtlseg_v1.0_sandbox \
        /bin/bash -c ./run_inference.sh

