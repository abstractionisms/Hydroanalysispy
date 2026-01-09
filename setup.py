"""
Setup configuration for hydrology package.

Install in development mode with:
    pip install -e .

Or install normally with:
    pip install .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding='utf-8') if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith('#')
        ]

# Separate optional dependencies
optional_requirements = {
    'dev': [
        'jupyterlab>=4.0.0',
        'pytest>=7.0.0',
        'black>=23.0.0',
        'flake8>=6.0.0',
    ],
    'dashboard': [
        'streamlit==1.34.0',
    ],
    'visualization': [
        'adjustText>=0.8.0',
    ],
}

setup(
    name="hydrology",
    version="1.0.0",
    author="Anonymous",
    description="Professional Python package for hydrological data analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abstractionisms/Hydroanalysispy",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require=optional_requirements,
    entry_points={
        'console_scripts': [
            'hydrology=hydrology.scripts.analyze_sites:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Hydrology",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="hydrology streamflow USGS NWIS climate analysis",
    project_urls={
        "Bug Reports": "https://github.com/abstractionisms/Hydroanalysispy/issues",
        "Source": "https://github.com/abstractionisms/Hydroanalysispy",
    },
)
