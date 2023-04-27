from .utils import initialize_gaussian_envelope, seed_everything
import torch
from models import ModelWrapper, CNNdense

class PytorchWrapper(ModelWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def compute_reg_loss(self, *args, **kwargs):
        return 0
    def prepare_regularization(self, normalize_reg=False):
        return 0
    def forward(self, x, pass_dict=False, *args, **kwargs):
        if type(x) is dict and not pass_dict:
            x = x['stim']
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim == 4:
            x = x.unsqueeze(1)
        return self.model(x, *args, **kwargs)

def get_cnn(config_init):
    def get_cnn_helper(config, device='cpu'):
        '''
            Create new model.
        '''
        seed_everything(config_init['seed'])
        num_layers = config['num_layers']
        num_filters = [config[f'num_filters{i}'] for i in range(num_layers)]
        filter_width = [config[f'filter_width{i}'] for i in range(num_layers)]
        # d2x = config['d2x']
        # d2t = config['d2t']
        # center = config['center']
        # edge_t = config['edge_t']
        scaffold = [len(num_filters)-1]
        num_inh = [0]*len(num_filters)
        if 'device' in config:
            device = config['device']
        # modifiers = {
        #     'stimlist': ['frame_tent', 'fixation_onset'],
        #     'gain': [deepcopy(drift_layer), deepcopy(sac_layer)],
        #     'offset': [deepcopy(drift_layer), deepcopy(sac_layer)],
        #     'stage': 'readout',
        # }

        scaffold = [len(num_filters)-1]
        num_inh = [0]*len(num_filters)
        modifiers = None
        
        input_dims = config_init['input_dims']
        cids = config_init['cids']
        mu = config_init['mu']
        
        cr0 = CNNdense(
                input_dims,
                num_filters,
                filter_width,
                scaffold,
                num_inh,
                is_temporal=False,
                NLtype='relu',
                batch_norm=True,
                norm_type=0,
                noise_sigma=0,
                NC=len(cids),
                bias=False,
                reg_core=None,
                reg_hidden=None,
                reg_readout={'glocalx':.1, 'l2':0.1},
                reg_vals_feat={'l1':0.01},
                cids=cids,
                modifiers=modifiers,
                window='hamming',
                device=device)

        # initialize parameters
        w_centered = initialize_gaussian_envelope( cr0.core[0].get_weights(to_reshape=False), cr0.core[0].filter_dims)
        cr0.core[0].weight.data = torch.tensor(w_centered, dtype=torch.float32)
        if mu is not None and hasattr(cr0.readout, 'mu'):
            cr0.readout.mu.data = torch.from_numpy(mu[cids].copy().astype('float32')).to(device)
            cr0.readout.mu.requires_grad = True
            cr0.readout.sigma.data.fill_(0.5)
            cr0.readout.sigma.requires_grad = True
        model = ModelWrapper(cr0)
        model.prepare_regularization()
        return model
    return get_cnn_helper

MODEL_DICT = {
    'CNNdense': get_cnn,
    'UNet': get_unet,
    # 'AttentionCNN': get_attention_cnn,
    'BioV': get_biov,
}

def verify_config(config):
    '''
        Convert config to the format supported by Ray.
    '''
    if 'filters' in config and 'kernels' in config:
        config.update({
            **{f'num_filters{i}': config['filters'][i] for i in range(len(config['filters']))},
            **{f'filter_width{i}': config['kernels'][i] for i in range(len(config['kernels']))},
            'num_layers': len(config['filters']),
        })
    
def get_model(config, factory=False):
    verify_config(config)
    if not factory:
        return MODEL_DICT[config['model']](config)(config)
    return MODEL_DICT[config['model']](config)

# def get_biov(config_init):
#     def get_biov_helper(config, device='cpu'):
#         seed_everything(config_init['seed'])
#         cids = config['cids']
#         input_dims = config_init['input_dims']
#         device = config_init['device']
#         pmodel = BioV(input_dims, cids, device)
#         pmodel.to(device)
#         return PytorchWrapper(pmodel)
#     return get_biov_helper

# def get_unet(config_init):
#     def get_unet_helper(config, device='cpu'):
#         seed_everything(config_init['seed'])
#         cids = config['cids']
#         model = ModelWrapper(UNet(cids))
#         model.to(device)
#         return model
#     return get_unet_helper

# def get_attention_cnn(config_init):
#     def get_attention_cnn_helper(config, device='cpu'):
#         cnn = get_cnn(config_init)(config, device)
#         cnn_out_dims = [cnn.model.core_subunits, *cnn.model.core[-1].output_dims[1:3], 1]
#         new_readout = LinearAttentionReadout(cnn_out_dims, config['cids'])
#         new_readout.to(device)
#         cnn.model.readout = new_readout         
#         return cnn
#     return get_attention_cnn_helper
