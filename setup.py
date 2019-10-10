import setuptools

exec(open("torchlayers/_version.py").read())  # get __version__

setuptools.setup(
    name="torchlayers",
    version=__version__,
    license="MIT",
    author="Szymon Maszke",
    author_email="szymon.maszke@protonmail.com",
    description="Shape inference and custom layers for PyTorch.",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/torchlayers",
    packages=setuptools.find_packages(),
    install_requires=["torch>=1.2.0"],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    project_urls={
        "Website": "https://szymonmaszke.github.io/torchlayers",
        "Documentation": "https://szymonmaszke.github.io/torchlayers/#torchlayers",
        "Issues": "https://github.com/szymonmaszke/torchlayers/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc",
    },
    keywords="machine learning analysis numpy pandas aws pipeline sklearn statsmodels",
)
