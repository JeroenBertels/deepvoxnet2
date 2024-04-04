import setuptools

with open("README.md", 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='deepvoxnet2',
    version='2.16.0',
    description='Yet another CNN framework: From pre- to postprocessing and keeping track of the spatial origin of the data.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/JeroenBertels/deepvoxnet2',
    author='Jeroen Bertels',
    author_email='jeroen.bertels@gmail.com',
    license_files=('LICENSE',),
    packages=setuptools.find_packages(),
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Operating System :: OS Independent"],
    python_requires='>=3.8, <3.10',
    install_requires=[
        'tensorflow>=2.4,<2.8',
        'tensorflow-addons>=0.11,<0.18',
        'tensorflow-probability>=0.11,<0.15',
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
        'scikit-learn',
        'xlrd>=2.0',
        'openpyxl>=3.0',
        'opencv-python>=4.5',
        'seaborn>=0.11.2',
        'comet-ml>=3.31.17',
        'twine>=4.0.1',
        'pymirc'
    ],
    extras_require={
        "sitk": [
            'simpleitk-simpleelastix',
        ]
    }
)
