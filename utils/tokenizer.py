#!/usr/bin/env python

import argparse
import glob
import logging
import os
import re
import sys

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import pandas as pd
import spacy
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
import string


logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s.%(funcName)s:%(lineno)d]'
                           ' %(message)s',
                    level=logging.INFO)

# parser = argparse.ArgumentParser()
# parser.add_argument(
#     '-v', '--verbose',
#     help="Be verbose",
#     default="NOTSET",
#     action="store_const",
#     dest="loglevel",
#      const=logging.INFO,
# )
# args = parser.parse_args()
LOG = logging.getLogger('proposal_scraper')

class PACManTokenizer(object):
    def __init__(self):

        # Load English tokenizer, tagger, parser, NER and word vectors
        self._spacy_nlp = spacy.load("en_core_web_sm")
        self._stop_words = None

    @property
    def flist(self):
        """List of files to process"""
        return self._flist

    @flist.setter
    def flist(self, value):
        self._flist = value

    @property
    def data_dict(self):
        """Python `dict` containing the text and category"""
        return self._data_df

    @data_dict.setter
    def data_dict(self, value):
        self._data_dict = value

    @property
    def data_df(self):
        """`pandas.DataFrame` containing text and category"""
        return self._data_df

    @data_df.setter
    def data_df(self, value):
        self._data_df = value

    @property
    def spacy_nlp(self):
        """spaCy Natural Language Processing object"""
        return self._spacy_nlp

    @spacy_nlp.setter
    def spacy_nlp(self, value):
        self._spacy_nlp = value

    @property
    def stop_words(self):
        return self._stop_words

    @stop_words.setter
    def stop_words(self, value):
        self._stop_words = value

    def get_stop_words(self,
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
        if default_stop_words is None:
            default_stop_words = self.spacy_nlp.Defaults.stop_words

        try:
            fobj = open(fname, 'r')
        except OSError as e:
            LOG.error(e)
        else:
            lines = fobj.readlines()
            custom_stop_words = set([line.strip('\n') for line in lines])

        # Use all the stop words defined by spaCy and the custom list in fname
        self.stop_words = default_stop_words.union(custom_stop_words)

    def spacy_tokenizer(self, text, stop_words=[], punctuations=[]):
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
        # Convert the text
        doc = self.spacy_nlp(text)

        mytokens = [token for token in doc]
        num_tokens = len(mytokens)

        # TODO: look into potential lemmatization issues with similar words used in different manners (e.g. galaxy and galactic)
        # Next, get the lemma of each token and force it to be lower case
        # mytokens = [
        #     word.lemma_.lower().strip('')
        #     if word.lemma_ != "-PRON-" else word.lower_
        #     for word in mytokens
        # ]

        mytokens = list(
            filter(lambda word: word.lemma_ != "-PRON-", mytokens)
        )
        lemmatize = lambda word: word.lemma_.lower().strip('')
        mytokens = list(map(lemmatize, mytokens))
        mytokens = list(
            filter(
                lambda word: word not in stop_words and
                             word not in punctuations,
                mytokens
            )
        )
        pattern = re.compile('[^a-zA-Z-]')
        mytokens = list(
            filter(
                lambda word: not pattern.match(word), mytokens
            )
        )

        return mytokens

    def run_tokenization(self, fname, N=20, plot=False):
        """ Tokenize the supplied file

        Parameters
        ----------
        fname : str
            Path to file containing the text to tokenize

        Returns
        -------

        """
        text, cleaned_text, tokens = None, None, None
        try:
            fobj = open(fname, 'r')
        except FileNotFoundError as e:
            LOG.error(e)
        else:
            text = fobj.readlines()
            text = [val.strip('\n') for val in text]
            text = ' '.join(text)
            tokens = self.spacy_tokenizer(
                text,
                stop_words=self.stop_words,
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


if __name__ == '__main__':
    p = PACManTokenizer()