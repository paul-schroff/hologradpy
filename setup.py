from setuptools import setup

name = 'hologradpy'
version = '1.0'
author = "Paul Schroff"
author_email = "paul.schroff@strath.ac.uk"
description = ("Module to holographically generete light "
               "potentials of arbitrary shape using a "
               "phase-modulating spatial light modulator (SLM).")
py_modules = ['hologradpy']
requirements = [
    'numpy',
    'scipy',
    'matplotlib',
    'torch',
    'kbnufft',
    'pytorch-minimize',
    'opencv-python',
    'checkerboard', 
    'slmsuite',
]

setup(name=name,
      version=version,
      author=author,
      author_email=author_email,
      description=description,
      py_modules=py_modules,
      install_requires=requirements,)
