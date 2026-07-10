from setuptools import setup, find_packages

setup(
    name="Computer-Vision-Model-Training-and-Deployment",
    version="0.1.0",
    author="Kandarpa Sarkar",
    author_email="kandarpaexe@gmail.com",
    description="Industry ready PyTorch training setup of ResNet",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kandarpa02/Computer-Vision-Model-Training-and-Deployment.git",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=['torch', 'numpy'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
    ],
    license="MIT",
    zip_safe=False,
    
)
