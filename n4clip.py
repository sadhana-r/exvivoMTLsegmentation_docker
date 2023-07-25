import SimpleITK as sitk
import sys
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str,
                    help="input filename")
parser.add_argument("-o", "--output", type=str,
                    help="output filename")
args = parser.parse_args()

#%%
inputImage = sitk.ReadImage(args.input, sitk.sitkFloat32)

#%%

corrector = sitk.N4BiasFieldCorrectionImageFilter()
shrinkFactor = 4
numberFittingLevels = 4

if shrinkFactor>1:
    image = sitk.Shrink(inputImage, [shrinkFactor] * inputImage.GetDimension())

corrector.SetMaximumNumberOfIterations([50] * numberFittingLevels)
corrected_image = corrector.Execute(image)

#%%

log_bias_field = corrector.GetLogBiasFieldAsImage(inputImage)
corrected_image_full_resolution = inputImage / sitk.Exp( log_bias_field )

#%%

sitk.WriteImage(sitk.Cast(corrected_image_full_resolution, sitk.sitkUInt16) ,
                args.output);
