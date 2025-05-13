from setuptools import find_packages, setup

setup(
    name="toyota",
    packages=find_packages(exclude=["toyota_tests"]),
    install_requires=[
        "dagster",
        "dagster-cloud"
    ],
    extras_require={"dev": ["dagster-webserver", "pytest"]},
)
