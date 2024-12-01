# NeuroVisKit (alpha release)
## Package for Neural Modeling and Analysis of The Visual System
We build on top of pytorch, allowing for the latest advancements in ML to be used for neuroscience.

1. See minimal.py for a minimal example of a modeling pipeline.
2. See tutorials for various tutorials and examples.

## Installation
### Environment (recommended)
It is recommended to create and activate a virtual environment using conda
```
conda create -n nvk python==3.11
conda activate nvk
```
First, install pytorch or make sure it is installed. Then install NeuroVisKit itself in one of three ways
1. From server (may be out of date)
```
python -m pip install NeuroVisKit
```
2. Directly from github
```
python -m pip install git+https://github.com/Yates-Lab/NeuroVisKit.git
```
3. From source (clone repo)
```
python -m pip install .
```

### Advance Installation
For installing the repo directly from github run `pip install git+https://github.com/Yates-Lab/NeuroVisKit.git`

(append @\<branch-name\>) as needed.

For installing an editable version, clone and then run 

```
cd NeuroVisKit
python -m pip install -e . -vvv
```

If imports don't work smoothly, go to VSCODE and search:
python -> pylance -> extra paths
and add the path to this dir
i.e. ~/Documents/NeuroVisKit
