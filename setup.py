# setup.py

"""Setup script for the QueryRefiner Python package."""

from setuptools import setup, find_packages

setup(
    name="queryrefiner",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "transformers>=4.30.0",
        "torch>=1.9.0",
        "datasets>=2.0.0",
    ],
    author="Yage Zhang",
    author_email="laqeey@gmail.com",
    description="A Python library for query refinement and code generation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/queryrefiner",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)