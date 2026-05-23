from setuptools import setup, find_packages

setup(
    name="embedding-debugger",
    version="0.1.0",
    description="Local-first toolkit for analyzing text embeddings, retrieval behavior, and robustness",
    author="Ke Lu",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "sentence-transformers>=2.7.0",
        "faiss-cpu>=1.8.0",
        "streamlit>=1.35.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "scikit-learn>=1.4.0",
        "umap-learn>=0.5.6",
        "plotly>=5.20.0",
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "tqdm>=4.66.0",
        "rich>=13.7.0",
        "tabulate>=0.9.0",
    ],
    entry_points={
        "console_scripts": [
            "embedding-debugger=app.streamlit_app:main",
        ]
    },
)
