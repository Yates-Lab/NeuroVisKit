import dill
import hashlib
import inspect
import torch
import io
from copy import deepcopy

from zmq import has
from NeuroVisKit._utils.lightning import clean_model_from_wandb

def hash_object_code(obj, hash_algorithm="shake128"):
    return hash_string(inspect.getsource(obj), hash_algorithm)

def hash_object_file(obj, hash_algorithm="shake128"):
    return hash_file(inspect.getsourcefile(obj), hash_algorithm)

def hash_string(string, hash_algorithm="sha256"):
    # Create a hash object using the specified algorithm
    hasher = hashlib.new(hash_algorithm)
    # Update the hash object with the string
    hasher.update(string.encode('utf-8'))
    # Return the hexadecimal representation of the hash
    if hash_algorithm == "shake128":
        return hasher.hexdigest(5)
    
    return hasher.hexdigest()
    
def hash_file(filename, hash_algorithm="sha256", buffer_size=65536):
    # Create a hash object using the specified algorithm
    hasher = hashlib.new(hash_algorithm)
    # Open the file in binary mode
    with open(filename, 'rb') as file:
        # Read the file in chunks and update the hash object
        while True:
            data = file.read(buffer_size)
            if not data:
                break
            hasher.update(data)
    # Return the hexadecimal representation of the hash
    if hash_algorithm == "shake128":
        return hasher.hexdigest(5)
    
    return hasher.hexdigest()

def dump(obj, file):
    obj = deepcopy(obj)
    #check if the object is a torch module
    try:
        if hasattr(obj, 'parameters') and next(obj.parameters()).device != 'cpu':
            obj.to("cpu")
        elif hasattr(obj, 'buffers') and next(obj.buffers()).device != 'cpu':
            obj.to("cpu")
        elif hasattr(obj, 'device'):
            obj = obj.to("cpu")
        clean_model_from_wandb(obj)
    except:
        print('could not move object to cpu during dump')
    with open(file, 'wb') as f:
        dill.dump(obj, f, byref=False, recurse=True)
        
class CPU_Unpickler(dill.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=torch.device('cpu'))
        else:
            return super().find_class(module, name)
        
def load(file):
    with open(file, 'rb') as f:
        return CPU_Unpickler(f, ignore=True).load()