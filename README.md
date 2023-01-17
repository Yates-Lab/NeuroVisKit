# foundation

To clone, run:
```console
git clone --recurse-submodules git@github.com:VisNeuroLab/foundation.git
```

Install Ray:
```python3
pip install -U "ray[default]"
pip install -U "ray[air]"
pip install -U "ray[tune]"
```

**For contributers:**

* In the datasets submodule, checkout the branch "foundation" and ensure its up to date.
* For each other submodule, checkout main.

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
* Run preprocess, then once it is done run schedule.