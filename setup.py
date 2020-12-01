import os
from setuptools import setup, find_packages

parent_dir = os.path.dirname(os.path.realpath(__file__))

with open(f"{parent_dir}/README.md", "r") as readme_file:
    long_description = readme_file.read()

setup(
    name="tfod_ssd",
    version="0.0.1",
    author="James Adams",
    author_email="monocongo@gmail.com",
    description="Object detection using TensorFlow/TensorRT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/monocongo/tfod_ssd",
    python_requires=">=3.6,<3.8",
    provides=[
        "tfod_ssd",
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
    ],
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        "arrow",
        "contextlib2",
        "Cython",
        "jupyter",
        "lxml",
        "matplotlib",
        "opencv-python",
        "numpy",
        "pillow",
        "tensorflow-gpu>=2.4.0",
    ],
    entry_points={
        "console_scripts": [
            "object_detect_images=tfod_ssd.detect_image:main",
            "object_detect_video=tfod_ssd.detect_video:main",
            "build_trt_graph=tfod_ssd.build_trt_graph:main",
        ]
    },
)
