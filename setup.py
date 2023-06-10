#!/usr/bin/env python
"""Setup for part_nerf"""
from distutils.extension import Extension
from itertools import dropwhile
from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import find_packages, setup


def collect_docstring(lines):
    """Return document docstring if it exists"""
    lines = dropwhile(lambda x: not x.startswith('"""'), lines)
    doc = ""
    for line in lines:
        doc += line
        if doc.endswith('"""\n'):
            break
    return doc[3:-4].replace("\r", "").replace("\n", " ")


def collect_metadata():
    meta = {}
    module_path = Path("src/part_nerf/__init__.py")
    with open(module_path) as f:
        lines = iter(f)
        meta["description"] = collect_docstring(lines)
        for line in lines:
            if line.startswith("__"):
                key, value = map(lambda x: x.strip(), line.split("="))
                meta[key[2:-2]] = value[1:-1]
    return meta


def get_extensions():
    return cythonize(
        [
            Extension(
                "part_nerf.external.libmesh.triangle_hash",
                sources=["src/part_nerf/external/libmesh/triangle_hash.pyx"],
                include_dirs=[np.get_include()],
                libraries=["m"],  # Unix-like specific
            ),
            Extension(
                "part_nerf.external.libmise.mise",
                sources=["src/part_nerf/external/libmise/mise.pyx"],
                include_dirs=[np.get_include()],
                libraries=["m"],  # Unix-like specific
            ),
            Extension(
                "part_nerf.external.libmcubes.mcubes",
                sources=[
                    "src/part_nerf/external/libmcubes/mcubes.pyx",
                    "src/part_nerf/external/libmcubes/pywrapper.cpp",
                    "src/part_nerf/external/libmcubes/marchingcubes.cpp",
                ],
                language="c++",
                include_dirs=[np.get_include()],
                extra_compile_args=["-std=c++11"],
                libraries=["m"],  # Unix-like specific
            ),
        ]
    )


def get_install_requirements():
    return [
        "numpy",
        "cython",
        "torch",
        "torchvision",
        "trimesh",
        "matplotlib",
        "Pillow",
        "pandas",
    ]


def get_long_description():
    with open("README.md") as f:
        long_description = f.read()
    return long_description


def setup_package():
    meta = collect_metadata()
    setup(
        name="part-nerf",
        version=meta["version"],
        description=meta["description"],
        long_description=get_long_description(),
        long_description_content_type="text/markdown",
        maintainer=meta["maintainer"],
        maintainer_email=meta["email"],
        url=meta["url"],
        keywords=meta["keywords"],
        license=meta["license"],
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "Topic :: Scientific/Engineering",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
        ],
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        install_requires=get_install_requirements(),
        ext_modules=get_extensions(),
    )


if __name__ == "__main__":
    setup_package()
