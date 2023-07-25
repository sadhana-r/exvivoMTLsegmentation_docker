# exvivoMTLsegmentation_docker
Docker for Laplace constrained MTL segmentation in PICSL 9.4T ex vivo MRI scans

# About

This is the set of files and scripts required to generate. The network is trained using a modified implementation of the nn-UNet framework which incorporated a Laplacian solver If using this tool to segment the MTL, please consider citing the following paper:

``
Ravikumar, S., Ittyerah, R., Lim, S., Xie, L., Das, S., Khandelwal, P., ... & Yushkevich, P. A. (2023, June). Improved Segmentation of Deep Sulci in Cortical Gray Matter Using a Deep Learning Framework Incorporating Laplace’s Equation. In International Conference on Information Processing in Medical Imaging (pp. 692-704). Cham: Springer Nature Switzerland.
``

This repository includes the Dockerfile and modified nnUnet scripts used for the. The trained model is saved on the cluster. The official container on DockerHub is labeled sadhanar/9.4exvivomtlseg:v1.0. 

The predicted multi-label segmentation consists of four labels: Gray Matter (1), White Matter (2) and Background (3) and SRLM (4). 

# How to generate segmentations

## Step 1: Download the docker image

     docker pull sadhanaravikumar/9.4exvivomtlseg:v1.0
 
## Step 2: Pre-process and prepare the data (Optional)

Before running inference, the data can be pre-processed and axis-aligned to faciliate manual editing. 

1. Run pre-processing script which applies N4 bias correction and intensity normalization
   
         docker run --user $(id -u) -it -v /path/to/inputdata/:/data/input sadhanaravikumar/9.4exvivomtlseg:v1.0 /bin/bash run_preprocessing.sh

2. Axis-align the scans

   a. Download the reference template image for axis-aigning the scans from [here](https://upenn.box.com/s/f4h0p96543dd3mx00bayag0u4s5iamap)
   
   b. Load the template image as the main image and MRI scan as an additional image. Enter manual registration mode and re-orient and alignn the scan to match the template image. More details on this process are provided [here](https://upenn.box.com/s/49co5uog6jl587tptan54pqnqdif9r0r)
   
   c. Save the transformation file and **make sure to save the resliced image with a filename ending in _0000.nii.gz**

## Step 3: Generation segmentation predictions

To generate the MTL segmentation, run the following command for inference on your data. Modify `/path/to/inputdata` to your directory where the input data ending in `*_0000.nii.gz` is located, and modify `/path/to/outputdata` to where you want to save the output predictions. The input and output data is located in `/data` inside the docker container. Leave the rest of the command as is.

    docker run --user $(id -u) -it --gpus all --privileged -v /path/to/inputdata/:/data/input -v /path/to/outputdata/:/data/output sadhanaravikumar/9.4exvivomtlseg:v1.0 /bin/bash run_inference.sh


# Sample Data

A sample dataset is located [here](https://upenn.box.com/s/zlj5r2pcvuqct5ynwf4znak3k3ky9jz3)
