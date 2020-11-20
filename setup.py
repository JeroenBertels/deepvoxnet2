from setuptools import setup


setup(
    name='deepvoxnet2',
    version='2.0.0',
    description='Deep learning processing framework for Keras.',
    url='https://github.com/JeroenBertels/deepvoxnet2',
    author='Jeroen Bertels, David Robben',
    author_email='jeroen.bertels@gmail.com',
    license='LGPL',
    packages=['deepvoxnet2'],
    zip_safe=False,
    install_requires=[
        'numpy>=1.15',
        'scipy>=1.1',
        'matplotlib>=2.2.2',
        'pydicom>=1.1',
        'scikit-image>=0.14',
        'numba>=0.39',
        'nibabel>=3.0'
    ],
    dependency_links=['https://github.com/gschramm/pymirc']
)
