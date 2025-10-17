import setuptools

with open("README.md", 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='deepvoxnet2',
    version='2.20.4',
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
    python_requires='>=3.12, <3.14',
    install_requires=[
        'tensorflow[and-cuda]==2.20.0',
        'nibabel==5.3.2',
        'scikit-learn==1.7.2',
        'pydot==4.0.1',
        'matplotlib==3.10.7',
        'numba==0.62.1',
        'pandas==2.3.3',  # optional; could be moved to extras_require later
        'transforms3d==0.4.2',  # optional; could be moved to extras_require later
    ],
    extras_require={
        "cupy": [
            'cupy-cuda12x==13.6.0',
        ],
        "full": [
            'cupy-cuda12x==13.6.0',
            'tensorflow-probability==0.25.0',
            'simpleitk-simpleelastix==2.5.0.dev49',
            'pydicom==3.0.1',
            'seaborn==0.13.2',
            'pymirc==0.30.2',
        ]
    }
)
