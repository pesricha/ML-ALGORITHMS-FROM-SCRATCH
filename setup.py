from setuptools import setup, find_packages

setup(
    name="ml_algo_lib",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "scikit-learn",  # For loading datasets
        "numpy",         # For numerical computations
        "matplotlib"     # For data visualization
    ],
)
