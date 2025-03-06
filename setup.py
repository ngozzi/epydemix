from setuptools import setup, find_packages

# Read the README file to use as the long description (optional)
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="epydemix",  
    version="0.1.0",  
    author="Nicolò Gozzi",  
    author_email="nic.gozzi@gmail.com",  
    description="A Python package for epidemic modeling, simulation, and calibration",  
    long_description=long_description,  
    long_description_content_type="text/markdown",  
    url="https://github.com/ngozzi/epydemix", 
    packages=find_packages(),  
    include_package_data=True,  
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # Specify the minimum Python version required
    install_requires=[
        "evalidate>=2.0.3",
        "matplotlib>=3.7.3",
        "numpy>=1.23.5",
        "pandas>=2.0.3",
        "scipy>=1.10.1",
        "seaborn>=0.13.2",
        "setuptools>=68.2.0"
    ],
    entry_points={
        'console_scripts': [
        ],
    },
)
