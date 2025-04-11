from pathlib import Path
from typing import List

import distutils
from distutils.command.install import install
from setuptools import setup, find_packages

PACKAGE_NAME = "vqa-service"
PACKAGE_VERSION = "0.0.1"
PACKAGE_URL = "https://github.com/cansik/vqa-service"

PACKAGE_DOC_MODULES = ["vqa"]

required_packages = find_packages(exclude=["tests", "examples", "scripts", "playground"])

BASE_NAME = "__required__"
ALL_NAME = "all"


def parse_requirements():
    extras = {}

    with open("requirements.txt") as f:
        lines = f.read().splitlines()

    extra_name = BASE_NAME
    extra_items: List[str] = []

    for line in [line.strip() for line in lines if line != ""]:
        if line.startswith("# extra"):
            extras[extra_name] = extra_items
            extra_items: List[str] = []

            tokens = line.split(" ")
            extra_name = tokens[2]

        elif line.startswith("#"):
            pass
        elif line.startswith("-"):
            pass
        else:
            extra_items.append(line)

    # add last group
    extras[extra_name] = extra_items

    # extract base packages
    install = extras.pop(BASE_NAME)

    # create all group
    all_reqs = list(extras.values())
    extras[ALL_NAME] = []
    for reqs in all_reqs:
        extras[ALL_NAME] += reqs

    return install, extras


install_required, extras_required = parse_requirements()

# read readme
current_dir = Path(__file__).parent
long_description = (current_dir / "README.md").read_text()

setup(
    name=PACKAGE_NAME,
    version=PACKAGE_VERSION,
    packages=required_packages,
    url=PACKAGE_URL,
    license="MIT License",
    author="Florian Bruggisser",
    author_email="github@broox.ch",
    description="VQA tools that allows users to ask questions about images and receive answers.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=install_required,
    extras_require=extras_required,
    entry_points={
        'console_scripts': [
            'vqa-service = vqa.__main__:main',
        ],
    },
)
