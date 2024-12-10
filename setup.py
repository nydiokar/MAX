# python/setup.py
from setuptools import setup, find_packages

setup(
    name="MAX",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "motor>=3.6.0",
        "pymongo>=4.9.2",
        "pydantic>=2.10.2",
        "anthropic>=0.32.0",
        "fastapi>=0.115.5",
        "pytest>=8.3.3",
        "pytest-asyncio>=0.24.0"
    ]
)