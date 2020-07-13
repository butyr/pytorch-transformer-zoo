import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytorch-transformer-butyr",
    version="0.1.0",
    package_dir={"": "src"},
    packages=setuptools.find_namespace_packages(where="src"),
    author="Leonid Butyrev",
    author_email="L.Butyrev@gmx.de",
    description="PyTorch transformer implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/butyr/pytorch-transformer",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'torch>=1.5.1',
        'tokenizers>=0.8.0',
        'tensorboard>=2.2.2',
    ],
)
