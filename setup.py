from setuptools import setup, find_packages
setup(
    name = 'scKinetics',
    version = '0.2.2',
    description = 'Biological prior guided single-cell kinetics inference.',
    author = 'Yanshuo Chen',
    author_email = 'poseidonchan@icloud.com',
    url = 'https://github.com/poseidonchan/scKinetics',
    license = 'GPL-3.0 license',
    packages = find_packages(),
    python_requires='>=3.9',
    platforms = 'any',
    install_requires = [
        'diffrax',
        'jax >= 0.4.13',
        'equinox >= 0.10.11',
        'optax',
        'anndata>=0.7.6',
        'tqdm',
        'matplotlib',
        'seaborn',
        'numpy',
    ],
)