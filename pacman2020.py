#!/usr/bin/env python

import glob
import logging
import os
import re


import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import pandas as pd
import spacy
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
# from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex
import string
import tqdm


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


def read_category_label(fname):

    flabel = fname.replace('Scientific_Justification','Scientific_Category')
    with open(flabel, 'r') as fobj:
        label = fobj.readlines()[0].strip().strip('\n')
    return label

def spacy_tokenizer(text, nlp=None, stop_words=[], punctuations=[]):
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
    if nlp is None:
        parser = English()
        doc = parser(text)
    else:
        doc = nlp(text)
    
    mytokens = [token for token in doc]
    num_tokens = len(mytokens)

    
# TODO: look into potential lemmatization issues with similar words used in different manners (e.g. galaxy and galactic)
    # Next, get the lemma of each token and force it to be lower case
    mytokens = [
        word.lemma_.lower().strip('')
        if word.lemma_ != "-PRON-" else word.lower_
        for word in mytokens
    ]

    # Removing stop words and punctuation
    mytokens = [
        word for word in mytokens
        if word not in stop_words and word not in punctuations
    ]
    # Keep anything that is a letter,number, or has a -
    # pattern = re.compile('[^a-zA-Z0-9-]')
    pattern = re.compile('[^a-zA-Z-]')
    mytokens = [word for word in mytokens if not pattern.match(word)]

    # print(f"Processed text represents "
    #       f"{100*len(mytokens) / num_tokens:0.2f}% of the input text")

    return mytokens


def tokenize(
    N=20,
    fname='/Users/nmiles/PACMan_dist/notebooks/test_sci_justification.txt',
    plot =False
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
    # nlp.tokenizer = custom_tokenizer(nlp)
    
    # Read in the Sci. Justif. Section
    with open(fname, 'r') as test_file:
        text = test_file.readlines()
        text = [val.strip('\n') for val in text]
        text = ' '.join(text)


    # Tokenize the example text
    # doc = nlp(text)
    # tokens = [token.text for token in doc if not token.is_stop and not token.is_space]
    tokens = spacy_tokenizer(
        text,
        nlp=nlp,
        stop_words=stop_words,
        punctuations=string.punctuation
    )
    cleaned_text = ' '.join(tokens)

    if plot:
        s = pd.Series(tokens)
        distribution = s.value_counts()
        fig, ax = plt.subplots(nrows=1, ncols=1)
        distribution[:N].plot.barh(ax=ax)
        ax.set_title(os.path.basename(fname))
        ax.set_xlim(0, 25)
        plt.show()

    return text, cleaned_text, tokens


def read_in_dataset(flist_label=None, flist_text=None, notebook=False):
    """
    Parameters
    ----------
    flist : None, optional
        Description
    """
    if flist_label is None:
        flist_label = glob.glob(
            '/Users/nmiles/PACMan_dist/C25/sci_just/*Category.txt'
        )

    if flist_text is None:
        flist_text = glob.glob(
            '/Users/nmiles/PACMan_dist/C25/sci_just/*Justification.txt'
        )

    if notebook:
        statusbar = tqdm.tqdm_notebook
    else:
        statusbar = tqdm.tqdm
    data = {'text':[], 'category':[]}
    LOG.info('Reading in dataset...')
    for f1, f2 in statusbar(zip(flist_label, flist_text)):
        label = read_category_label(f1)
        data['category'].append(label)
        text, cleaned_text, tokens = tokenize(fname=f2, plot=False)
        data['text'].append(text)

    df = pd.DataFrame(data)
    return df, data




if __name__ == '__main__':
    import glob
    flist = glob.glob('/Users/nmiles/PACMan_dist/C25/sci_just/*txt')
    idx = np.random.randint(0, len(flist))
    tokenize(fname=flist[idx])