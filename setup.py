from setuptools import setup, find_packages

setup(
    name="womd-torch-metrics",
    version="0.1.0",
    description="Waymo Open Motion Dataset metrics for PyTorch",
    author="",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
        ],
    },
    python_requires=">=3.7",
)

