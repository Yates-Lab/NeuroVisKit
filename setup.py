#!/usr/bin/env python
import setuptools

if __name__ == "__main__":
    setuptools.setup(
        name='NeuroVisKit',
        version="0.0.1",
        description='NeuroVisKit',
        author='Dekel Galor',
        packages=[
            'NeuroVisKit',
            'NeuroVisKit.tests',
            'NeuroVisKit.utils',
            'NeuroVisKit._utils',
            # 'NeuroVisKit._future'
        ],
        package_dir={
            'NeuroVisKit': '.',
            'NeuroVisKit.tests': 'tests',
            'NeuroVisKit.utils': 'utils',
            'NeuroVisKit._utils': '_utils',
            # 'NeuroVisKit._future': '_future'
        },
        install_requires=[
            'numpy',
            'tqdm',
            'lab',
            'ipykernel',
            'dill',
            'wandb',
            'imageio',
            'lightning',
            'matplotlib',
            'scipy',
            'scikit-image',
            'IPython'
        ]
    )