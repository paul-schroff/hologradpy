from setuptools import setup

name = 'hologradpy'
version = '1.0'
author = "Paul Schroff"
author_email = "paul.schroff@strath.ac.uk"
description = ("Module to holographically generete light "
               "potentials of arbitrary shape using a "
               "phase-modulating spatial light modulator (SLM).")
requirements = [
    'numpy',
    'scipy',
    'matplotlib',
    'torch',
    'pytorch-minimize',
    'opencv-python',
    'checkerboard', ]

setup(name=name,
      version=version,
      author=author,
      author_email=author_email,
      description=description,
      install_requires=requirements,)
