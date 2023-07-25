#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch
from torch import nn
import numpy as np
import nibabel as nib
from nnunet.utilities.nd_softmax import softmax_helper
import torch.nn.functional as F


class MultipleOutputLoss2(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss2, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, x, y):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        l = weights[0] * self.loss(x[0], y[0])
        for i in range(1, len(x)):
            if weights[i] != 0:
                l += weights[i] * self.loss(x[i], y[i])
        return l

#Added by SR for computing DICE loss with SOR module
class MultipleOutputLoss2_SOR_DSC_only(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        :param lambda_weight: weighting between segmentation loss and thickness loss
        """
        super(MultipleOutputLoss2_SOR_DSC_only, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

        self.bnd = 15

    def forward(self, x, y):

        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"


        #Extract thickness prediction from x, convert to seg and compute dice loss with ground truth laplacian
        # Need to mask the hippocampus region out of the predicted laplacian map
        gm_mask = y[1]

        #Mask the target incase of interpolation errors
        r = np.random.randint(10)

        #One hot encoded laplacian segmentation
        t_predseg = x[1]

        #Already a segmentation - converted in augmentaion function
        t_gt = y[0]

        #Convert WM/CSF regions to label 0 in ground truth. Loss computation ignores regions where ground truth is 0 from dice comp
        t_gt[gm_mask != 1] = 0

        #Crop the boundaries of laplacian maps due to boundary errors
        t_gt = t_gt[:,:,self.bnd:-self.bnd, self.bnd:-self.bnd, self.bnd:-self.bnd]
        t_predseg = t_predseg[:,:,self.bnd:-self.bnd, self.bnd:-self.bnd, self.bnd:-self.bnd]

        # Compute loss over pixels included in the gm.
        t_loss = self.loss(t_predseg,t_gt)

        loss_array = torch.tensor([0, t_loss])

        return t_loss, loss_array

#Added by SR for computing DICE loss with SOR module
class MultipleOutputLoss2_SOR_DSC(nn.Module):
    def __init__(self, loss, weight_factors=None, sor_start_epoch = 10, lambda_weight = 0.5):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        :param lambda_weight: weighting between segmentation loss and thickness loss
        """
        super(MultipleOutputLoss2_SOR_DSC, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss
        self.lambda_weight = lambda_weight

        self.sor_start_epoch = sor_start_epoch
        
        self.bnd = 15
    
    def update_lambda(self, lambda_weight):
        self.lambda_weight = lambda_weight

    def forward(self, x, y, epoch):

        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"


        #Extract thickness prediction from x and compute mse loss with ground truth laplacian
        # Need to mask the hippocampus region out of the predicted laplacian map
        if epoch > self.sor_start_epoch:

            gm_mask = y[1]

            #Mask the target incase of interpolation errors
            #r = np.random.randint(10)

            #Convert to one hot encoded segmentation (6 channels)
            t_predseg = x[1] 

            #Already a segmentation - converted in augmentaion function
            t_gt = y[0]

            #Convert WM/CSF regions to label 0 in ground truth. Loss computation ignores regions where ground truth is 0 from dice comp
            t_gt[gm_mask != 1] = 0

            #Also remove background from laplacian prediction
            laplace_mask = t_gt < 1 # True for background region
            laplace_mask =  laplace_mask.repeat(1,t_predseg.shape[1],1,1,1)
            t_predseg[laplace_mask] = 0

            #Crop the boundaries of laplacian maps due to boundary errors
            t_gt = t_gt[:,:,self.bnd:-self.bnd, self.bnd:-self.bnd, self.bnd:-self.bnd]
            t_predseg = t_predseg[:,:,self.bnd:-self.bnd, self.bnd:-self.bnd, self.bnd:-self.bnd]

            """
            #For checking outputs - save a few patch outputs
            output_file = '/data/sadhanar/groundtruth_exp6_seg' + str(r) + '.nii.gz'
            laplace = torch.swapaxes(t_gt.squeeze(),0,3)
            nib.save(nib.Nifti1Image(laplace.cpu().numpy(), np.eye(4)),output_file)

            output_file = '/data/sadhanar/pred_exp6_seg' + str(r) + '.nii.gz'
            laplace = torch.swapaxes(t_predseg.argmax(1).squeeze(),0,3)
            nib.save(nib.Nifti1Image(laplace.detach().cpu().numpy(), np.eye(4)),output_file)
            """

            # Compute DCandCE loss over pixels included in the gm.
            t_loss = self.loss(t_predseg,t_gt)
            
            x = x[2:]

        # First element is the ground truth laplacian map/seg
        y = y[1:]

        if self.weight_factors is None:
            weights = [1] * (len(x) - 1) # since first element in x is the thickness map
        else:
            weights = self.weight_factors

        # Convert the hippocampus label in the ground truth to gray matter for dice loss
        gt = y[0]
        gt[gt == 5] = 1

        l = weights[0] * self.loss(x[0], gt)

        for i in range(1, len(x)):
            # Idexing of weights and y is off by one because I already removed the laplacian map
            if weights[i] != 0:
                gt = y[i]
                gt[gt == 5] = 1
                l += weights[i] * self.loss(x[i], gt)

        #Return either both loss summed together or just segmentation loss
        if epoch > self.sor_start_epoch:
            loss_array = torch.tensor([l, t_loss])
            #l = l + t_loss
            l = (1- self.lambda_weight)*l + self.lambda_weight*t_loss  # for weighted experiment, t_loss is upweighted
        else:
            loss_array = torch.tensor([l, 0])

        return l, loss_array

#Added by SR for computing MSE loss with SOR module
class MultipleOutputLoss2_SOR(nn.Module):
    def __init__(self, loss, weight_factors=None, sor_start_epoch = 10, lambda_weight = 0.5):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        :param lambda_weight: weighting between segmentation loss and thickness loss
        """
        super(MultipleOutputLoss2_SOR, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss
        self.lambda_weight = lambda_weight

        #MSE loss - first try without averaging over number of gm pixels
        self.mse_loss = nn.MSELoss()
        self.sor_start_epoch = sor_start_epoch

        self.bnd = 15

    def update_lambda(self, lambda_weight):
        self.lambda_weight = lambda_weight

    def forward(self, x, y, epoch):

        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"


        #Extract thickness prediction from x and compute mse loss with ground truth laplacian
        # Need to mask the hippocampus region out of the predicted laplacian map
        if epoch > self.sor_start_epoch:
            gm_mask = y[1]

            #Mask the target incase of interpolation errors
            r = np.random.randint(10)

            #Only include elements in the ground truth gm
            t_pred = x[0]
            t_pred[gm_mask != 1] = 0

            t_gt = y[0]
            t_gt[gm_mask != 1] = 0

            #Mask by ground truth segmentation
            t_pred = t_pred[gm_mask == 1]
            t_gt = t_gt[gm_mask==1]

            # Compute average MSE loss over pixels included in the gm.
            t_loss = self.lambda_weight*self.mse_loss(t_pred,t_gt)

            x = x[2:]

        # First element is the ground truth laplacian map
        y = y[1:]

        if self.weight_factors is None:
            weights = [1] * (len(x) - 1) # since first element in x is the thickness map
        else:
            weights = self.weight_factors

        # Convert the hippocampus label in the ground truth to gray matter for dice loss
        gt = y[0]
        gt[gt == 5] = 1

        l = weights[0] * self.loss(x[0], gt)

        for i in range(1, len(x)):
            # Idexing of weights and y is off by one because I already removed the laplacian map
            if weights[i] != 0:
                gt = y[i]
                gt[gt == 5] = 1
                l += weights[i] * self.loss(x[i], gt)

        #Extract thickness prediction from x and compute mse loss with ground truth laplacian
        # Need to mask the hippocampus region out of the predicted laplacian map
        if epoch > self.sor_start_epoch:
            loss_array = torch.tensor([l, t_loss])
            l = l + t_loss
        else:
            loss_array = torch.tensor([l, 0])

        return l, loss_array
