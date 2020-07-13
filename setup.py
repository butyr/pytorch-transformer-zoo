import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytorch-transformer-butyr",
    version="0.0.1",
    author="Leonid Butyrev",
    author_email="L.Butyrev@gmx.de",
    description="pytorch transformer implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/butyr/pytorch-transformer",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)

