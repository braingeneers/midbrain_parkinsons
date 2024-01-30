from setuptools import setup, find_packages

setup(
    name="Human_Hippocampus",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Collaborative Environment for Analyzing the Human Hippocampus datasets",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    # Add other necessary information and parameters
)