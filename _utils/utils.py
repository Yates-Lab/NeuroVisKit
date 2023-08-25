import sys, getopt, os, gc
import numpy as np
import random
import torch

def joinCWD(*paths):
    return os.path.join(os.getcwd(), *paths)

def isInteractive():
    """Check if the current session is interactive or not."""
    #can use __name__  != "__main__" or hasattr(__main__, 'get_ipython')
    return bool(getattr(sys, 'ps1', sys.flags.interactive))

def get_opt_dict(opt_config, default=None):
    """
        Get the requested options from command line as a dictionary
        opt_config: takes in a list of tuples, where each tuple is one of four options:
        A. options that take an argument from the command line
            1. (shorthand:, fullname=, func) applies a function on the argument
            2. (shorthand:, fullname=) gives the argument as a string
        B. options that do not take an argument from the command line
            3. (shorthand, fullname, value) sets argument to value if the option is present
            4. (shorthand, fullname) sets argument to True if the option is present
    """
    argv = sys.argv[1:]
    opt_names = [i[1] for i in opt_config if i[1]]
    opt_shorthand = [i[0] for i in opt_config if i[0]]
    opts, args = getopt.getopt(argv, "".join(opt_shorthand), opt_names)
    opts_dict = {}
    opt_names = ['--'+str(i[1]).replace('=', '') for i in opt_config]
    opt_shorthand = ['-'+str(i[0]).replace(':', '') for i in opt_config]
    for opt, arg in opts:
        opt_tuple = opt_config[opt_names.index(opt) if opt in opt_names else opt_shorthand.index(opt)]
        opt_name = (opt_tuple[1] or opt_tuple[0])
        if '=' in opt_name or ':' in opt_name:
            opt_func = opt_tuple[2] if len(opt_tuple) > 2 else lambda x: x
            opts_dict[opt_name.replace('=', '').replace(':', '')] = (opt_func or (lambda x: x))(arg)
        else:
            opts_dict[opt_name] = opt_tuple[2] if len(opt_tuple) > 2 else True
    if default is not None:
        default.update(opts_dict)
        return default
    return opts_dict

def seed_everything(seed, noprint=False):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    torch.cuda.manual_seed(str(seed))
    if not noprint:
        print(f'Seeding with {seed}')

def memory_clear():
    '''
        Clear unneeded memory.
    '''
    torch.cuda.empty_cache()
    gc.collect()