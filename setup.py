from setuptools import setup, find_packages

setup(
    name="phot-gal",
    version="0.1.0",
    packages=find_packages(),
    include_package_data = True,
    platforms="any",
    setup_requires=[
        'numpy; python_version>="3.6"',
    ],
    install_requires=[
        'scipy; python_version>="3.6"',
        'scikit-learn',
        'eazy',
        'ngboost'
    ],
    project_urls={
        'Source': 'https://github.com/DhruvZ/Phot-Gal',
    },
)
