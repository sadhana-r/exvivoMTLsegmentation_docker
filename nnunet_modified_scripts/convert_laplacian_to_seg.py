import torch

def doublesigmoid_threshold(data, lower_lim, upper_lim):

    steepness = 10

    lower_thresh = 1/(1 + torch.exp(-steepness*(data - lower_lim)))
    upper_thresh = 1/(1 + torch.exp(steepness*(data - upper_lim)))

    output = torch.mul(lower_thresh,upper_thresh)
    output = output.squeeze()

    return output

def convert_laplacian_toseg(data):

    #thresholds = [-0.2,0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.95] 
    thresholds = [-0.3,0, 0.25,0.5,0.75,0.95]
    shape = (data.shape[0],len(thresholds), *data.shape[2:])
    result = torch.zeros(shape, dtype=data.dtype)
    for i, l in enumerate(thresholds):
        output = doublesigmoid_threshold(data, l,l+0.3)
        result[:,i] = output

    return result

def convert_laplacian_toseg_finer(data):

    thresholds = [-0.2,0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.95]
    shape = (data.shape[0],len(thresholds), *data.shape[2:])
    result = torch.zeros(shape, dtype=data.dtype)
    for i, l in enumerate(thresholds):
        output = doublesigmoid_threshold(data, l,l+0.3)
        result[:,i] = output

    return result

