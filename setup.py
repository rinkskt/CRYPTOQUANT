from setuptools import setup, find_packages

setup(
    name="crypto-quant",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "streamlit",
        "pandas",
        "sqlalchemy",
        "requests"
    ],
)