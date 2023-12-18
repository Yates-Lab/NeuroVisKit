import torch

def irfC(x, model, neuron_inds=None, batched=False):
    '''Computes the IRFs for given neuron inds, time contiguous model and input
        
        x (batched=False): input to the model shaped (num_lags, *spatial_dims)
        x (batched=True): input to the model shaped (batch_size, num_lags, *spatial_dims)
        model: model to compute IRFs for
        neuron_inds: list of neuron indices to compute IRFs for
    '''
    if neuron_inds is None:
        neuron_inds = slice(None)
        
    if not batched:
        def f(inp):
            inp = inp[0] #inp is a list of inputs, we only want the first one
            pred = model(inp)[:, neuron_inds]
            return pred[-1] #we care about the last prediction
        return torch.autograd.functional.jacobian(f, x.unsqueeze(0)).squeeze(0)
    else:
        model = torch.vmap(model, in_dims=0, out_dims=0)
        def f(inp):
            pred = model(inp)[:, :, neuron_inds]
            return pred[:, -1] #we care about the last prediction
        return torch.autograd.functional.jacobian(f, x)
    
def irf(x, model, neuron_inds=None):
    """Computes the IRFs for given neuron inds, input, and traditional batched model.

    Args:
        x (Tensor): input to the model shaped (batch_size, *any_shape)
        model (nn.Module): model to compute IRFs for.
        neuron_inds: list of neuron indices to compute IRFs for.
    """
    if neuron_inds is None:
        neuron_inds = slice(None)
    def f(inp):
        pred = model(inp)[:, neuron_inds]
        return pred
    return torch.autograd.functional.jacobian(f, x)