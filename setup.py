# setup.py
from setuptools import find_packages, setup

setup(
    name="agrisignal",
    version="1.0.0",
    packages=find_packages(include=["agrisignal", "agrisignal.*"]),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "requests>=2.31.0",
        "pandera>=0.17.0",
        "xgboost>=2.0.0",
        "scikit-learn>=1.3.0",
        "yfinance>=0.2.28",
        "prefect>=2.13.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "prometheus-client>=0.19.0",
        "pydantic>=2.5.0",
        "pyyaml>=6.0",
        "mlflow>=2.9.0",
        "shap>=0.44.0",
    ],
    python_requires=">=3.10",
)
