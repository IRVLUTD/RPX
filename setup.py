import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rpx-benchmark",
    version="0.0.1",
    author="IRVLUTD",
    description="RPX: Robot Perception Benchmark for Deployment Conditions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pillow",
        "matplotlib",
        "tqdm",
    ],
    extras_require={
        "torch": ["torch", "torchvision"],
        "profiling": ["fvcore", "thop", "scipy"],
        "all": ["torch", "torchvision", "fvcore", "thop", "scipy"],
    },
)
