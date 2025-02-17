# python/setup.py
from setuptools import setup, find_packages

setup(
    name="max",
    version="0.2.2",
    packages=find_packages(),
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