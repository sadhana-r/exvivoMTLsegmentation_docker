# FROM ubuntu:latest
FROM nvidia/cuda:11.3.1-runtime-ubuntu18.04

#To disable installation of optional dependencies
RUN echo 'APT::Install-Suggests "0";' >> /etc/apt/apt.conf.d/00-docker
RUN echo 'APT::Install-Recommends "0";' >> /etc/apt/apt.conf.d/00-docker

#Install python but in non interactive mode and clean up package list
#Need to have git installed to install nU-Net
RUN DEBIAN_FRONTEND=noninteractive \
  apt update \
  && apt install -y python3.8 python3-setuptools python3-pip python3-venv git \
  && rm -rf /var/lib/apt/lists/* \
  && python3 -m pip install --upgrade pip

#Create a virtual environement
ENV VIRTUAL_ENV=/tmp/nnunet_env 
RUN pip3 install virtualenv
RUN virtualenv -p python3 ${VIRTUAL_ENV}

#Install requirements for nnUnet
COPY . /tmp/

#Set workdirectory
WORKDIR /tmp/

ENV MOD_SCRIPTS=/tmp/nnunet_modified_scripts
ENV NNUNET_DIR=${VIRTUAL_ENV}/src/nnunet/nnunet

## Needed to upgrade pip and specify torch version for CUDA compatability. This needs to be in one docker layer so I can copy 
## the modified nnunet scripts. 
RUN /bin/bash -c "source ${VIRTUAL_ENV}/bin/activate \
    && python3 -m pip install --no-cache-dir --upgrade pip \
    && pip install torch==1.10.1 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 \
    && pip3 install -r /tmp/requirements.txt \
    && cp ${MOD_SCRIPTS}/neural_network_SOR.py $NNUNET_DIR/network_architecture/neural_network_SOR.py \
    && cp $MOD_SCRIPTS/generic_UNet_SOR.py $NNUNET_DIR/network_architecture/generic_UNet_SOR.py \
    && cp $MOD_SCRIPTS/move_datachannel_to_seg.py $NNUNET_DIR/training/data_augmentation/move_datachannel_to_seg.py \
    && cp $MOD_SCRIPTS/data_augmentation_moreDA_SOR.py $NNUNET_DIR/training/data_augmentation/data_augmentation_moreDA_SOR.py \
    && cp $MOD_SCRIPTS/downsampling_SOR.py $NNUNET_DIR/training/data_augmentation/downsampling_SOR.py \
    && cp $MOD_SCRIPTS/dataset_loading_SOR.py  $NNUNET_DIR/training/dataloading/dataset_loading_SOR.py \
    && cp $MOD_SCRIPTS/deep_supervision.py $NNUNET_DIR/training/loss_functions/deep_supervision.py \
    && cp $MOD_SCRIPTS/convert_laplacian_to_seg.py $NNUNET_DIR/utilities/convert_laplacian_to_seg.py \
    && cp $MOD_SCRIPTS/nnUNetTrainerV2_SOR_MTLatlas.py $NNUNET_DIR/training/network_training/nnUNetTrainerV2_SOR_MTLatlas.py \
    && cp $MOD_SCRIPTS/nnUNetTrainerV2_SORseg_exp6_fixedaug_run2.py $NNUNET_DIR/training/network_training/nnUNetTrainerV2_SORseg_exp6_fixedaug_run2.py"