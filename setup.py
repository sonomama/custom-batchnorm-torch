from setuptools import find_packages, setup

setup(
    name="custom-batchnorm",
    packages=find_packages(),
    version="0.1.0",
    extras_require={"dev": ["pre-commit", "pytest", "flake8", "black"]},
    license="MIT",
)
