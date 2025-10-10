import setuptools

with open("README.md", 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='deepvoxnet2',
    version='2.20.2',
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
        'transforms3d==0.4.2',
        'pandas==2.3.3',
        'scikit-learn==1.7.2',
        'cupy-cuda12x==13.6.0',
        'pydot==4.0.1',
        'matplotlib==3.10.7',
        'numba==0.62.1'
    ],
    extras_require={
        "full": [
            'tensorflow-probability',
            'simpleitk-simpleelastix',
            'pydicom',
            'numba',
            'seaborn',
            'pymirc',
        ]
    }
)
