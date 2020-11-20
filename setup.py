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
        'tensorflow>=2.3.1',
        'tensorflow-addons>=0.11.2',
        'scipy>=1.5.2',
        'nibabel>=3.1.1',
        'pydicom>=2.0.0',
        'numba>=0.39.0',
        'matplotlib>=2.2.2',
        'scikit-learn>=0.20.2',
        'pandas>=0.24.1'
    ],
    dependency_links=['https://github.com/gschramm/pymirc']
)
