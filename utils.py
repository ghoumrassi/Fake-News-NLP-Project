import re
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.models.keyedvectors import KeyedVectors
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from textblob import TextBlob


# FIX ENCODING
def get_special_char_prevalence(
    df: pd.DataFrame, text_columns: List, count_col: str
) -> pd.DataFrame:
    # Concatenate the text columns into a single series
    all_text = df[text_columns].apply(lambda x: ' '.join(x), axis=1)
    # Remove alphanumeric characters and whitespace to isolate special
    # characters
    all_special_chars = (all_text
                         .str.replace(r'[\w\s]', '', regex=True)
                         .str.cat())

    char_dict = {char: all_special_chars.count(
        char) for char in set(all_special_chars)}
    char_df = pd.DataFrame(list(char_dict.items()),
                           columns=['Char', count_col])

    return char_df


def standardize_text(text: str) -> str:
    replacements = {
        '’': "'",    # Replace ’ with '
        '“': '"',    # Replace “ with "
        '”': '"',    # Replace ” with "
        '‘': "'",    # Replace ‘ with '
    }

    translation_table = str.maketrans(replacements)
    standardized_text = text.translate(translation_table)

    return standardized_text


# FEATURE ENGINEERING

def get_capitalised(text: str) -> float:
    text_words = text.split()
    if len(text_words) == 0:
        return 0.0

    return sum(map(str.isupper, text_words)) / len(text_words)


def get_symbol_prop(
    text: str, symbols: Union[str, Tuple, List], regex: bool = False
) -> float:
    if regex:
        if isinstance(symbols, Union[Tuple, List]):
            raise ValueError(
                '`symbols` must be of type `str` if `regex==True`.')
        return len(re.findall(symbols, text)) / len(text)
    else:
        if isinstance(symbols, str):
            symbols = [symbols]

        return sum([text.count(symbol) for symbol in symbols]) / len(text)


def get_sent_subj_score(text: str) -> pd.Series:
    tb = TextBlob(text).sentiment
    return pd.Series([tb.polarity, tb.subjectivity])


# Get the embedding for each token and take the mean to get a document
# embedding
def get_embeddings(text: str, model: KeyedVectors) -> pd.Series:
    tokens = word_tokenize(text)

    # Load stopwords set
    stop_words = set(stopwords.words('english'))

    tokens = [token for token in tokens if token not in stop_words]

    word_embeddings = [model[token]
                       for token in tokens if token in model]

    if len(word_embeddings) == 0:
        return pd.Series([0.0] * 25)

    embeddings = np.mean(word_embeddings, axis=0)
    return pd.Series(embeddings)


# FEATURE ENGINEERING - VISUALISATION
def plot_histograms(
    df: pd.DataFrame, fields_to_plot: List[str], height_per_plot: float = 3.5,
    log_x_scale: bool = False
) -> None:

    num_plots = len(fields_to_plot)
    fig_height = num_plots * height_per_plot

    fig, ax = plt.subplots(num_plots, 2, figsize=(12, fig_height))

    for i, field in enumerate(fields_to_plot):

        for j, plot_type in enumerate(["stack", "fill"]):

            # Plot histogram
            if log_x_scale:
                sns.histplot(data=df, x=field, hue="IsFake",
                             multiple=plot_type, ax=ax[i, j],
                             log_scale=(True, False))
            else:
                sns.histplot(data=df, x=field, hue="IsFake",
                             multiple=plot_type, ax=ax[i, j])

            # Set titles and labels
            title = (("Proportional " if plot_type == 'fill' else '')
                     + f'Distribution of `{field}` by Fake/Real Status')

            ylabel = 'Proportion' if plot_type == 'fill' else 'Frequency'

            ax[i, j].set_title(title)
            ax[i, j].set_xlabel(field)
            ax[i, j].set_ylabel(ylabel)

    # Adjust layout
    plt.tight_layout(pad=3.0)
    plt.show()


# MODEL EVALUATION
def get_binary_clf_eval_metrics(
    y_test: np.ndarray, y_pred: np.ndarray
) -> Dict[str, Union[float, np.ndarray]]:
    # Calculate various evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    bal_accuracy = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print the evaluation metrics
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Balanced Accuracy: {bal_accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return {
        "accuracy": float(accuracy),
        "bal_accuracy": float(bal_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }
