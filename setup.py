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
    python_requires='>=3.6, <3.9',
    install_requires=[
        'tensorflow>=2.3',
        'tensorflow-addons>=0.11',
        'numpy>=1.15,<1.19',
        'scipy>=1.5',
        'nibabel>=3.1',
        'pydicom>=2.0',
        'numba>=0.39',
        'matplotlib>=2.2.2',
        'scikit-image>=0.14'
    ],
    dependency_links=[
        'https://github.com/gschramm/pymirc'
    ]
)
