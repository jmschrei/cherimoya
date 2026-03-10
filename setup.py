from setuptools import setup

setup(
    name='cherimoya',
    version='0.0.1',
    author='Jacob Schreiber',
    author_email='jmschreiber91@gmail.com',
    packages=['cherimoya'],
    scripts=['cli/cherimoya'],
    url='https://github.com/jmschrei/cherimoya',
    license='LICENSE.txt',
    description='A light-weight deep learning architecture that uses modern optimization tricks to achieve strong predictive performance.',
    install_requires=[
        "numpy >= 1.14.2",
        "scipy >= 1.0.0",
        "pandas >= 1.3.3",
        "torch >= 1.9.0",
        "h5py >= 3.7.0",
        "tqdm >= 4.64.1",
        "seaborn >= 0.11.2",
        "modisco >= 2.0.0",
        "tangermeme >= 0.2.3",
		"macs3",
        "bam2bw",
		"bpnet-lite"
    ],
)
