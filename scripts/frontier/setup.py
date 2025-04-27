from setuptools import setup, find_packages

# Information about the package
setup(
    name="matensemble",  # Package name
    version="0.1.0",  # Initial version
    author="Soumendu Bagchi",  # Author's name
    author_email="soumendubagchi@gmail.com",  # Author's email
    description="An adaptive and highly asynchronous ensemble simulation workflow manager matEnsemble \
(https://github.com/BagchiS6/matensemble; built jointly on top of the hierarchical graph based \
scheduler FLUX and concurrent-futures infrastructure of python",  # A short description
    long_description=open('README.md').read(),  # Detailed description (typically from a README file)
    long_description_content_type="text/markdown",  # Format for long description (Markdown)
    url="https://github.com/BagchiS6/matensemble",  # URL for the package repository (GitHub, etc.)
    packages=find_packages(),  # Automatically find and include all packages
    classifiers=[  # Classifiers to categorize the package
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
    ],
    install_requires=["numpy>=1.19.2", "pandas>=1.1.0"
    ],
    python_requires='>=3.11',  # Minimum Python version required
    # entry_points={  # Optional: Define entry points for scripts (if applicable)
    #     'console_scripts': [
    #         'xyz-cli=xyz.cli:main',  # Example if there's a command-line tool (xyz-cli)
    #     ],
    # },
)
