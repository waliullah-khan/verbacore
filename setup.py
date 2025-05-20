from setuptools import setup, find_packages

setup(
    name="text-analysis",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy==1.24.3",
        "pandas==2.0.3",
        "matplotlib==3.7.2",
        "seaborn==0.12.2",
        "torch==2.0.1",
        "transformers==4.31.0",
        "sentence-transformers==2.2.2",
        "scikit-learn==1.3.0",
        "nltk==3.8.1",
        "gensim==4.3.1",
        "pyLDAvis==3.4.1",
        "vaderSentiment==3.3.2",
        "textblob==0.17.1",
        "umap-learn==0.5.3",
        "hdbscan==0.8.29",
        "plotly==5.15.0",
        "prophet==1.1.4",
        "tqdm==4.65.0",
        "jupyter==1.0.0"
    ]
) 