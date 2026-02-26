from setuptools import setup, find_packages

setup(
    name="match-outcome-predictor",
    version="0.1.0",
    description="ML + NLP system for football match outcome prediction",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.1.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "transformers>=4.33.0",
        "torch>=2.0.0",
        "streamlit>=1.28.0",
        "newsapi-python>=0.2.7",
    ],
)
