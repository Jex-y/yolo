import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="yolo-tf2",
    version="1.0.0",
    description="Expandable and flexable yolo model library built using tensorflow 2",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/Jex-y/yolo",
    author="Edward Jex",
    author_email="edwardjex@live.co.uk",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["yolo-tf2"],
    include_package_data=True,
    install_requires=["tensorflow", "numpy", "python-opencv", "PyYAML"],
    },
)