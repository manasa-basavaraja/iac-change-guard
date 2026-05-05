from setuptools import find_packages, setup

setup(
    name="iac-change-guard",
    version="0.1.0",
    description="IaC unified diff scoring with trees, distilbert, attention",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "PyYAML>=6.0.1",
        "numpy>=1.24.0",
        "tqdm>=4.66.0",
        "python-hcl2>=4.3.0",
    ],
    extras_require={"dev": ["pytest>=7.4.0"]},
)
