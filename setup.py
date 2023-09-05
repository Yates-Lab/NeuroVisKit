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
        ],
        package_dir={
            'NeuroVisKit': 'NeuroVisKit',
            'NeuroVisKit.tests': 'tests',
            'NeuroVisKit.utils': 'NeuroVisKit/utils',
            'NeuroVisKit._utils': 'NeuroVisKit/_utils',
        },
        # package_data={
        #     'NeuroVisKit': ['*'],
        # }
    )