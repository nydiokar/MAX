# python/setup.py
from setuptools import setup, find_packages

setup(
    name="multi_agent_orchestrator",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "motor",
        "chromadb",
        "pytest",
        "pytest-asyncio"
    ]
)