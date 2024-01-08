# Fake News Detection

```
Author: Adam Ghoumrassi
Est. Completion Time: 6-7 hours
```

This project involved building machine learning models to detect fake news articles. The goals were to:

- Achieve high predictive accuracy in classifying articles as real or fake
- Understand which features are most indicative of fake vs real news
- Build interpretable models to explain predictions
- Optimise model inference speed for production deployment

## Compute and Inference Time

Careful consideration was taken to ensure that the code is well optimised and does not utilise complex models unnecessarily.

For example:

- GloVe-25 embeddings were used in favour of larger embeddings such as Word2Vec or BERT.
- Embeddings and sentiment scores were only generated for article titles, rather than the entire contents.
- LightGBM was favoured over deep learning based systems.

The final script takes in the order of **3 minutes** to run on an *AMD Ryzen 9 5900HS* CPU, and delivers a model that is capable of inference speeds of around **0.0014ms per article**.

Additionally, the GloVe embedding download can take several minutes depending on internet speeds.

## Data

The dataset contained 25,473 news articles labeled as either real or fake. It included the article title, text contents, subject category, and publication date.
Methods

The following steps were taken:

- **Data cleaning:** Removed duplicates, standardised text encodings
- **Feature engineering:** Extracted features related to capitalisation, punctuation, sentiment, etc. that may indicate fake news
- **Modeling:** Trained Logistic Regression and LightGBM models using cross-validation grid/randomised search for hyperparameter tuning
- **Evaluation:** Compared model accuracy, precision, recall, F1 score, ROC curves, etc.
- **Explainability:** Analysed model coefficients and feature importances to understand detection patterns
- **Inference speed:** Benchmarked throughput to assess production readiness

## Key Results

- LightGBM achieved best accuracy of 99.9% and F1 score of 0.98 (only five misclassified records)
- Most indicative fake news patterns related to sentiment, text length, and symbol usage
- Logistic regression had 3.3x faster inference, useful for low-latency production apps

## Feature Engineering

A combination of engineered features and document embeddings were used as the input features for this solution.

Details of engineered features:

- Day of week article was posted
- Frequency of capitalised words
- Frequency of symbol usage in the text / title (see notebook for further details)
- Sentiment / Subjectivity score of the title

For document embedding, GloVe-25 embeddings were utilised with mean pooling to convert from word- to document-level embeddings.

## Model Architecture

The solution utilises both logistic regression and LightGBM models. Logistic regression was chosen for its interpretability and fast inference time. LightGBM was chosen for its high predictive accuracy from complex feature interactions.

By evaluating both types of models, we can balance accuracy versus explainability depending on the use case. Logistic regression provides direct visibility into feature coefficients. LightGBM reaches higher F1 performance but has reduced explainability.

The decision was taken to avoid deep learning based approaches for this challenge, given the computational complexity running such systems at scale and that the solution already achieves high predictive accuracy.

## Next Steps

Potential ways to improve the models:

- Further Exploratory Analysis: n-grams, topic modelling, etc.
- Bayesian Search for more sophisticated hyperparameter optimisation.
- Test performance with a wider range of classifier types (SVM, Naive Bayes, Neural Networks).
- Recursive Feature Elimination - remove features that may not be improving generality.
- Further feature engineering - misspellings, expletives, readability, named entities, etc.
- Experiment with embedding type and pooling (Word2vec, BERT, OpenAI Ada)
- Investigate Feature Drift - is future data following the same patterns as training data.
- Investigate potential bias in data collection process.
- More detailed investigation of failures.

## Directory Structure

| Path             | Description                                                           |
| ---------------- | --------------------------------------------------------------------- |
| data/raw         | Contains the input data CSV files.                                    |
| enviroment.yml   | Environment specification for package installation through Conda.     |
| README.md        | You're already here!                                                  |
| requirements.txt | List of Pip packages for installation without Conda.                  |
| FakeNews.ipynb   | The main notebook - contain analysis, model training, and evaluation. |
| utils.py         | Additional Python utilities for boilerplate functions.                |

## Setup Instructions

To properly run the submission.ipynb notebook, you need to have a Conda environment set up. Please ensure that you have the Conda package manager installed beforehand. The instructions provided below were tested with Conda `22.11.1`.

### Creating and Activating the Environment with Conda

To create and activate a new environment using the provided environment.yml file, follow these steps:

1. Open your terminal (or an Anaconda Prompt if you are on Windows).
2. Navigate to the directory containing the environment.yml file.
3. Execute the following commands to create and activate the environment:

   - `conda env create -f environment.yml`
   - `conda activate adamghoumrassi_env`

By running these commands, Conda will install all necessary dependencies and activate the new environment named adamghoumrassi_env.

### Alternative Setup Using venv

If you prefer not to use Conda, you can alternatively set up the environment using venv along with `requirements.txt`.

1. Open your terminal.
2. Navigate to the directory containing the requirements.txt file.
3. Execute the following commands to create a virtual environment and install dependencies:
   - `python -m venv adamghoumrassi_env`
   - `source adamghoumrassi_env/bin/activate`  # On Windows, use `'adamghoumrassi_env\Scripts\activate'`
   - `pip install -r requirements.txt`

### Running the Notebook

Once the environment is activated, you can run the submission.ipynb Jupyter notebook. Open your terminal or Anaconda Prompt, ensure the adamghoumrassi_env environment is activated, and then start Jupyter Notebook or JupyterLab.

Navigate to the submission.ipynb file in the Jupyter interface and open it. You should now be able to run the cells within the notebook.

Please ensure that you deactivate your environment after you have finished working with `conda deactivate`.
