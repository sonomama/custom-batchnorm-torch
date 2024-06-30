from setuptools import find_packages, setup

setup(
    name="custom-batchnorm",
    packages=find_packages(),
    version="0.1.0",
    install_requires=["torch==2.2.1+cu121"],
    extras_require={"dev": ["pre-commit", "pytest", "flake8", "black"]},
    license="MIT",
)
