"""Setup script for Game of Life Mosaics package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gol-mosaics",
    version="2.0.0",
    author="Michiel Rollier",
    description="Create artistic mosaics from images using Conway's Game of Life still-life patterns",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mrollier/game-of-life-mosaics",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Artistic Software",
        "Topic :: Scientific/Engineering :: Artificial Life",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[req for req in requirements if not req.startswith("pytest")],
    extras_require={
        "dev": ["pytest>=7.0.0", "pytest-cov>=3.0.0"],
        "bg-removal": ["rembg[cpu]>=2.0.0"],
    },
    include_package_data=True,
    package_data={
        "gol_mosaics": ["../../data/*.npy"],
    },
    zip_safe=False,
)
