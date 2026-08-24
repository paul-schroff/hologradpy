from setuptools import find_packages, setup

name = "hologradpy"
version = "1.0"
author = "Paul Schroff"
author_email = "paul.schroff@strath.ac.uk"
description = (
    "Accurate SLM holography from a calibrated, differentiable model of the optical "
    "setup."
)
license = "LGPL-3.0-only"
url = "https://github.com/paul-schroff/hologradpy"
requirements = [
    "numpy >= 1.26",
    "scipy",
    "matplotlib",
    "torch",
    "pytorch-minimize",
    "opencv-python",
    "checkerboard",
    "pillow",
    "slmsuite",
    "tqdm",
    "asdf",
    "einops",
    "array_api_compat",
    "jaxtyping",
    "kornia", # TODO: We can probably do without kornia.
]

extras = {
    "nufft": ["torchkbnufft"],
}

setup(
    name=name,
    version=version,
    author=author,
    author_email=author_email,
    description=description,
    license=license,
    url=url,
    packages=find_packages(include=["hologradpy", "hologradpy.*"]),
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require=extras,
)
