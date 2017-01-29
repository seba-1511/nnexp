#!/usr/bin/env python

from setuptools import (
        setup as install,
        find_packages,
        )

VERSION = '0.0.1'

install(
        name='nnexp',
        packages=['nnexp'],
        version=VERSION,
        description='Neural net experiments for beginners',
        author='Seb Arnold',
        author_email='smr.arnold@gmail.com',
        url = 'https://github.com/seba-1511/nnexp',
        download_url = 'https://github.com/seba-1511/nnexp/archive/0.1.3.zip',
        license='License :: OSI Approved :: Apache Software License',
        classifiers=[],
        scripts=[
            ]
)
