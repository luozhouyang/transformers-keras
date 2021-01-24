import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="transformers_keras",
    version="0.2.2",
    description="Transformer-based models implemented in tensorflow 2.x(Keras)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/luozhouyang/transformers-keras",
    author="ZhouYang Luo",
    author_email="zhouyang.luo@gmail.com",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
    ],
    extras_require={
        "tf": ["tensorflow>=2.0.1"],
        "tf-gpu": ["tensorflow-gpu>=2.0.1"],
    },
    license="Apache Software License",
    classifiers=(
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    )
)
