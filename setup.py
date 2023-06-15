# -*- coding: utf-8 -*-
import os

import setuptools

setuptools.setup(
    name='utils',
    use_scm_version=True,  # 使用 setuptools-scm 做自動版號的控管 (需 # pip install setuptools-scm)
    keywords='demo',
    description='A demo for python packaging.',
    long_description=open(
        os.path.join(
            os.path.dirname(__file__),
            'README.md'
        ), encoding="utf-8"
    ).read(),
    author='deryannhuang',
    author_email='deryann.huang@auo.com',
    url='http://tcaigitlab.corpnet.auo.com/mfg/adtea3/smartmeasure.git',
    packages=setuptools.find_packages(),
    license='MIT'
)
