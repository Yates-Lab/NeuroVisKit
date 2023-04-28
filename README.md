# foundation

To clone, run:
```console
git clone --recurse-submodules git@github.com:VisNeuroLab/foundation.git
cd foundation
git submodule update --init --recursive --remote
```

To pull, run:
```console
git pull --recurse-submodules
```

Install Ray:
```python3
pip install -U "ray[default]"
pip install -U "ray[air]"
pip install -U "ray[tune]"
```

*make sure to modify your datadir in preprocess.py to the location of the raw datasets.

**For contributers:**

Run the following script after cloning:
```console
cd datasets
git checkout -b foundation
git pull origin foundation

cd ../NDNT
git checkout main
cd ../models
git checkout main
cd ../
```

**Sanity Check**:
* Run preprocess, then once it is done run train.py
