import dill
import hashlib
import inspect
from copy import deepcopy

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
        elif hasattr(obj, 'device'):
            obj = obj.to("cpu")
    except:
        print('could not move object to cpu during dump')
    with open(file, 'wb') as f:
        dill.dump(obj, f, byref=False, recurse=True)
        
def load(file):
    with open(file, 'rb') as f:
        return dill.load(f, ignore=True)