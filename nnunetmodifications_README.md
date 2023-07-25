Scripts that have been modiied from the original nnUnet for SOR training

1. Network Trainer
/data/sadhanar/nnUNet/nnUNet/nnunet/training/network_training/nnUNetTrainerV2_SOR_MTLatlas.py
/data/sadhanar/nnUNet/nnUNet/nnunet/training/network_training/nnUNetTrainerV2_SORseg_exp6_fixedaug_run2.py for full model

2. Network Architecture
/data/sadhanar/nnUNet/nnUNet/nnunet/network_architecture/neural_network_SOR.py
/data/sadhanar/nnUNet/nnUNet/nnunet/network_architecture/generic_UNet_SOR.py

3. Data Augmentation
/data/sadhanar/nnUNet/nnUNet/nnunet/training/data_augmentation/move_datachannel_to_seg.py 
/data/sadhanar/nnUNet/nnUNet/nnunet/training/data_augmentation/data_augmentation/data_augmentation_moreDA_SOR.py
/data/sadhanar/nnUNet/nnUNet/nnunet/training/data_augmentation/data_augmentation/downsampling_SOR.py

4. Data Loading
/data/sadhanar/nnUNet/nnUNet/nnunet/training/dataloading/dataset_loading_SOR.py 

5. Loss Function
/data/sadhanar/nnUNet/nnUNet/nnunet/training/loss_functions/deep_supervision.py

6. Utilities
/data/sadhanar/nnUNet/nnUNet/nnunet/utilities/convert_laplacian_to_seg.py