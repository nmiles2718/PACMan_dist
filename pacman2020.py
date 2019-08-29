#!/usr/bin/env python

import logging

from astropy.io import fits

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import pandas as pd
import spacy
from spacy.lang.en import English
import string


logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s.%(funcName)s:%(lineno)d]'
                           ' %(message)s')
LOG = logging.getLogger('pacman2020')
LOG.setLevel(logging.INFO)


def read_stop_words(
        fname='/Users/nmiles/PACMan_dist/libs/stopwords.txt',
        default_stop_words=None
):
    """ Read in custom list of stop words stored in the input file

    Parameters
    ----------
    fname : str
        full path to file containing a list of stop words

    default_stop_words : list
        List of the default stop words using by the NLP toolkit

    Returns
    -------

    """

    with open(fname, 'r') as test_file:
        text = test_file.readlines()
        stop_words = set([val.strip('\n') for val in text])

    # Compare the stop words to the defaults used in spaCy
    missing_stop_words = stop_words.difference(default_stop_words)

    if len(missing_stop_words) !=0:
        # These are both sets, so we take the union (i.e. we 'or' them)
        default_stop_words |= set(missing_stop_words)

    return default_stop_words


def spacy_tokenizer(text, stop_words=[], punctuations=[]):
    """ Tokenizer using the spaCy nlp toolkit.

    Parameters
    ----------
    text : str
        A block of text in a single string
    stop_words : list
        A list of the stop words to filter out

    punctuations : list
        A list of the punctuations to filter out

    Returns
    -------
    mytokens : list
        A list of all the tokens found after removing stop words and punctuation

    """
    # Creating tokenize the input next
    parser = English()
    mytokens = parser(text)
    num_tokens = len(mytokens)

# TODO: look into potential lemmatization issues with similar words used in different manners (e.g. galaxy and galactic)
    # Next, lemmatize each token and standardize the capitalization to be lower case
    mytokens = [
        word.lemma_.lower().strip()
        if word.lemma_ != "-PRON-" else word.lower_
        for word in mytokens
    ]

    # Removing stop words and punctuation
    mytokens = [
        word for word in mytokens
        if word not in stop_words and word not in punctuations
    ]
    # print(f"Processed text represents "
    #       f"{100*len(mytokens) / num_tokens:0.2f}% of the input text")

    return mytokens


def tokenize(
        N=20,
        fname='/Users/nmiles/PACMan_dist/notebooks/test_sci_justification.txt',

):
    """ TODO: FINISH UP THIS PORTION

    Parameters
    ----------
    N : int
        Integer to use for displaying the top N most common words

    fname : str
        file to process

    Returns
    -------

    """

    nlp = spacy.load("en_core_web_sm")
    stop_words = read_stop_words(
        fname='/Users/nmiles/PACMan_dist/libs/stopwords.txt',
        default_stop_words=nlp.Defaults.stop_words
    )
    # Reset the default list to contain all the words from spaCy stop word list
    # and our custom stop word list
    nlp.Defaults.stop_words = stop_words

    # Read in the Sci. Justif. Section
    with open(fname, 'r') as test_file:
        text = test_file.readlines()
        text = [val.strip('\n') for val in text]
        text = ' '.join(text)

    # Tokenize the example text
    tokens = spacy_tokenizer(
        text,
        stop_words=stop_words,
        punctuations=string.punctuation
    )
    s = pd.Series(tokens)
    distribution = s.value_counts()
    distribution[:N].plot.barh()
    plt.show()
    return text, tokens


if __name__ == '__main__':
    tokenize()