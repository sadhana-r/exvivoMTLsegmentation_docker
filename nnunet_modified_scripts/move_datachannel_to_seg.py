# Written by Sadhana Ravikumar
# Script used for running successive over relaxation model


import torch
from batchgenerators.transforms.abstract_transforms import AbstractTransform
import numpy as np
from nnunet.utilities.convert_laplacian_to_seg import convert_laplacian_toseg, convert_laplacian_toseg_finer
import nibabel as nib


class MoveLaplaceToSeg(AbstractTransform):
        '''
        data_dict['output_key'] will be a list of segmentations scaled according to ds_scales
        '''
        def __init__(self,channel_id, key_origin="data", key_target="seg", remove_from_origin = True, laplace_seg = False):
            self.remove_from_origin = remove_from_origin
            self.key_target = key_target
            self.key_origin = key_origin
            self.channel_id = channel_id
            self.laplace_seg = laplace_seg

        def __call__(self, **data_dict):

            origin = data_dict.get(self.key_origin)
            target = data_dict.get(self.key_target)
           

            laplace = origin[:,self.channel_id]
            #laplace = target[:,0]
            #target = target[:,1]
            #target = target[:,None,:]

            r = np.random.randint(10)
            
            laplace = laplace[:,None,:]

            # Currently only using this mode
            if self.laplace_seg:
                #Convert to seg one hot
                laplace = torch.from_numpy(laplace)
                laplace_onehot = convert_laplacian_toseg_finer(laplace).numpy()
                laplace_multilabel = laplace_onehot.argmax(1)
                laplace_multilabel = laplace_multilabel[:,None, :]
                target = np.concatenate((target, laplace_multilabel), 1)

            # This won't work anymore - can't feed laplacian map to segmentation augmentation functions. To fix (April 2022)
            else: 
                target = np.concatenate((target, laplace), 1)
            
            data_dict[self.key_target] = target

            if self.remove_from_origin:
                remaining_channels = [i for i in range(origin.shape[1]) if i != self.channel_id]
                origin = origin[:, remaining_channels]
                data_dict[self.key_origin] = origin
            
            return data_dict

