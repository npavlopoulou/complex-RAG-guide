"""
Helper Functions for Controllable RAG System

This module contains utility functions for text processing, document manipulation,
PDF handling, similarity analysis, and metric evaluation for RAG applications.
"""

# Standard library imports
import re
import textwrap

# Third-party imports
import tiktoken
import PyPDF2
import pylcs
import pandas as pd
import dill
from langchain.docstore.document import Document

from src.configuration.constants import Constants


# =============================================================================
# TEXT PROCESSING FUNCTIONS
# =============================================================================

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """
    Calculates the number of tokens in a given string using a specified encoding.

    Args:
        string (str): The input string to tokenize.
        encoding_name (str): The name of the encoding to use (e.g., 'cl100k_base').

    Returns:
        int: The number of tokens in the string according to the specified encoding.
    """
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def replace_double_lines_with_one_line(text):
    """
    Replaces consecutive double newline characters ('\n\n') with a single newline character ('\n').

    Args:
        text (str): The input text string.

    Returns:
        str: The text string with double newlines replaced by single newlines.
    """
    cleaned_text = re.sub(r'\n\n', '\n', text)
    return cleaned_text


def escape_quotes(text):
    """
    Escapes both single and double quotes in a string.

    Args:
        text (str): The string to escape.

    Returns:
        str: The string with single and double quotes escaped.
    """
    return text.replace('"', '\\"').replace("'", "\\'")


def text_wrap(text, width=120):
    """
    Wraps the input text to the specified width.

    Args:
        text (str): The input text to wrap.
        width (int, optional): The width at which to wrap the text. Defaults to 120.

    Returns:
        str: The wrapped text.
    """
    return textwrap.fill(text, width=width)

# =============================================================================
# SIMILARITY AND ANALYSIS FUNCTIONS
# =============================================================================

def is_similarity_ratio_lower_than_th(large_string, short_string, th):
    """
    Checks if the similarity ratio between two strings is lower than a given threshold.

    Uses the Longest Common Subsequence (LCS) algorithm to calculate similarity.

    Args:
        large_string (str): The larger string to compare.
        short_string (str): The shorter string to compare.
        th (float): The similarity threshold (0.0 to 1.0).

    Returns:
        bool: True if the similarity ratio is lower than the threshold, False otherwise.
    """
    # Calculate the length of the longest common subsequence (LCS)
    lcs = pylcs.lcs_sequence_length(large_string, short_string)

    # Calculate the similarity ratio
    similarity_ratio = lcs / len(short_string)

    # Check if the similarity ratio is lower than the threshold
    return similarity_ratio < th


def analyse_metric_results(results_df):
    """
    Analyzes and prints the results of various RAG evaluation metrics.

    Args:
        results_df (pandas.DataFrame): A pandas DataFrame containing the metric results.
    """
    metric_descriptions = {
        "faithfulness": "Measures how well the generated answer is supported by the retrieved documents.",
        "answer_relevancy": "Measures how relevant the generated answer is to the question.",
        "context_precision": "Measures the proportion of retrieved documents that are actually relevant.",
        "context_relevancy": "Measures how relevant the retrieved documents are to the question.",
        "context_recall": "Measures the proportion of relevant documents that are successfully retrieved.",
        "context_entity_recall": "Measures the proportion of relevant entities mentioned in the question that are also found in the retrieved documents.",
        "answer_similarity": "Measures the semantic similarity between the generated answer and the ground truth answer.",
        "answer_correctness": "Measures whether the generated answer is factually correct."
    }

    for metric_name, metric_value in results_df.items():
        print(f"\n**{metric_name.upper()}**")

        # Extract the numerical value from the Series object
        if isinstance(metric_value, pd.Series):
            metric_value = metric_value.values[0]

        # Print explanation and score for each metric
        if metric_name in metric_descriptions:
            print(metric_descriptions[metric_name])
            print(f"Score: {metric_value:.4f}")
        else:
            print(f"Score: {metric_value:.4f}")


# =============================================================================
# OBJECT SERIALIZATION FUNCTIONS
# =============================================================================

def save_object(obj, filename):
    """
    Save a Python object to a file using dill serialization.
    
    Args:
        obj: The Python object to save.
        filename (str): The name of the file where the object will be saved.
    """
    with open(filename, 'wb') as file:
        dill.dump(obj, file)
    print(f"Object has been saved to '{filename}'.")


def load_object(filename):
    """
    Load a Python object from a file using dill deserialization.
    
    Args:
        filename (str): The name of the file from which the object will be loaded.
    
    Returns:
        object: The loaded Python object.
    """
    with open(filename, 'rb') as file:
        obj = dill.load(file)
    print(f"Object has been loaded from '{filename}'.")
    return obj


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
# save_object(plan_and_execute_app, 'plan_and_execute_app.pkl')
# plan_and_execute_app = load_object('plan_and_execute_app.pkl')
