import sys
import setuptools


setuptools.setup(
    name='deepvoxnet2',
    version='2.10.4',
    description='Deep learning processing framework for Keras.',
    url='https://github.com/JeroenBertels/deepvoxnet2',
    author='Jeroen Bertels, David Robben',
    author_email='jeroen.bertels@gmail.com',
    license='LGPL',
    packages=setuptools.find_packages(),
    zip_safe=False,
    python_requires='>=3.6, <3.10',
    install_requires=[
        'tensorflow>=2.3,<2.8',
        'tensorflow-addons>=0.11',
        'tensorflow-probability>=0.11',
        'numpy>=1.15',
        'scipy>=1.5',
        'nibabel>=3.1',
        'pydicom>=2.0',
        'numba>=0.39',
        'matplotlib>=2.2.2',
        'scikit-image>=0.14',
        'transforms3d>=0.3.1',
        'jupyter>=1.0',
        'Pillow>=8.1.0',
        'pandas>=1.2',
        'sklearn',
        'xlrd>=2.0',
        'openpyxl>=3.0',
        'pymirc'
    ],
    extras_require={
        "sitk": [
            'simpleitk-elastix' if sys.version_info < (3, 9) else 'simpleitk-simpleelastix',
        ]
    }
)
