#!/usr/bin/env python

from collections import defaultdict
import glob
import logging
import os
import re
import time

import dask
from dask.diagnostics import ProgressBar
import joblib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import tqdm
from utils.tokenizer import PACManTokenizer


logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s.%(funcName)s:%(lineno)d]'
                           ' %(message)s')
LOG = logging.getLogger('pacman2020')
LOG.setLevel(logging.INFO)


class PACManPipeline(PACManTokenizer):
    def __init__(self, model_name=None):
        super().__init__()
        self._base = os.path.join(
            '/',
            *os.path.dirname(os.path.abspath(__file__)).split('/')
        )

        self._encoder = LabelEncoder()
        self._model_name = model_name
        self._model = None


    @property
    def base(self):
        return self._base

    @base.setter
    def base(self, value):
        self._base = value

    @property
    def encoder(self):
        return self._encoder

    @encoder.setter
    def encoder(self, value):
        self._encoder = value

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, value):
        self._model_name = value

    def load_model(self, model_name=None, encoder_name=None):
        LOG.info(f"Loading model stored at \n {model_name}")
        if model_name is not None:
            self.model_name = model_name
        self.model = joblib.load(self.model_name)

        LOG.info(f"Loading encoder information stored at \n {encoder_name}")

    def predict(self, X):
        self.predictions = self.model.predict(X)
        self.predicition_probabilities = self.model.predict_proba(X)

    def preprocess(self, flist, parallel=False):
        """ Read in the dataset and perform the necessary pre-processing steps
        Parameters
        ----------
        flist : list
            Description
        """

        if self.stop_words is None:
            self.get_stop_words(fname=self.stop_words_file)

        st = time.time()
        # data = {'text':[], 'cleaned_text':[], 'fname':[]}
        data = defaultdict(list)

        if parallel:
            delayed_obj = [
                dask.delayed(self.run_tokenization)(fname=f, plot=False)
                for f in flist
            ]
            with ProgressBar():
                results = dask.compute(*delayed_obj, scheduler='threads')
        else:
            # with ProgressBar():
                results = [
                    self.run_tokenization(fname=f, plot=False)
                    for f in tqdm.tqdm(flist)
                ]

        for i, (text, cleaned_text, tokens) in enumerate(results):
            data['text'].append(text)
            data['cleaned_text'].append(cleaned_text)
            data['fname'].append(flist[i])
            data['proposal_num'].append(
                int(flist[i].split('/')[-1].split('_')[0])
            )
        df = pd.DataFrame(data)
        et = time.time()
        duration = (et - st)/60
        LOG.info(f"Total time for preprocessing: {duration:.3f}")
        return df

    def read_data(self, cycle, parallel=False):
        path_to_data = os.path.join(
            self.training_set,
            f"training_corpus_cy{cycle}"
        )

        flist = glob.glob(
            f"{path_to_data}/*training.txt")

        N = len(flist)

        LOG.info(
            (f"Reading in {N} proposals...\n"
             f"Data Directory: {path_to_data}")
        )

        df = self.preprocess(flist=flist, parallel=parallel)




class PACManTrain(PACManPipeline):

    def __init__(self, cycles_to_analyze=[24, 25]):
        PACManPipeline.__init__(self)
        PACManTokenizer.__init__(self)
        self.training_set = os.path.join(
            self.base,
            'training_data'
        )
        self.cycles_to_analyze = cycles_to_analyze
        self.proposal_data = {}
        self.encoders = {}
        self.stop_words_file = os.path.join(
            self.base,
            'utils',
            'stopwords.txt'
        )

    def read_training_data(self, parallel=False, N=None):
        """ Initilaize everything for training

        Returns
        -------
        """


        # First, read in our custom list of stop words using the
        # get_stop_words() method of the PACManTokenizer object


        for cycle in self.cycles_to_analyze:
            path_to_data = os.path.join(
                self.training_set,
                f"training_corpus_cy{cycle}"
            )

            flist = glob.glob(
                f"{path_to_data}/*training.txt")

            N = len(flist)

            LOG.info(
                (f"Reading in {N} proposals...\n"
                 f"Data Directory: {path_to_data}")
            )

            df = self.preprocess(flist=flist, parallel=parallel)

            hand_classifications = pd.read_csv(
                f"{path_to_data}/cycle_{cycle}_hand_classifications.txt"
            )
            merged_df = pd.merge(df, hand_classifications, on='proposal_num')
            self.proposal_data[f"cycle_{cycle}"] = merged_df
            labels = self.encoder.fit_transform(
                self.proposal_data[f"cycle_{cycle}"]['hand_classification']
            )

            self.proposal_data[f"cycle_{cycle}"]['encoded_hand_classification']\
                = labels
            N=None

    def fit_model(self, df, clf=None, vect=None):
        if clf is None:
            clf = MultinomialNB(alpha=0.05)
        if vect is None:
            vect = TfidfVectorizer(
                max_features=10000,
                use_idf=True,
                norm='l2',
                ngram_range=(1, 2)
            )

        self.model = Pipeline(
            [('vect', vect),
             ('clf', clf)]
        )
        self.model.fit(df['cleaned_text'], df['encoded_hand_classification'])

    def save_model(self, fname=None):
        if fname is not None:
            fout = os.path.join(
                self.base,
                'models',
                fname
            )
        else:
            fout = os.path.join(
                self.base,
                'models',
                'pacman_model.joblib'
            )

        joblib.dump(self.model, fout)



def read_in_dataset(flist, parallel=False):
    """ Read in the dataset and perform the necessary pre-processing steps
    Parameters
    ----------
    flist : list
        Description
    """
    LOG.info(f"Reading in {len(flist)} proposals...")

    st = time.time()
    data = {'text':[], 'cleaned_text':[], 'fname':[]}
    pacman = PACManTokenizer()
    pacman.get_stop_words()
    if parallel:
        delayed_obj = [
            dask.delayed(pacman.run_tokenization)(fname=f, plot=False)
            for f in flist
        ]
        with ProgressBar():
            results = dask.compute(*delayed_obj, scheduler='threads')
    else:
        with ProgressBar():
            results = [
                pacman.run_tokenization(fname=f, plot=False)
                for f in tqdm.tqdm(flist)
            ]

    for i, (text, cleaned_text, tokens) in enumerate(results):
        data['text'].append(text)
        data['cleaned_text'].append(cleaned_text)
        data['fname'].append(flist[i])
    df = pd.DataFrame(data)
    et = time.time()
    duration = (et - st)/60
    LOG.info(f"Total time for preprocessing: {duration:.3f}")
    return df

def train_classifier(model, model_params, training_df):

    tfidf_vect = TfidfVectorizer(
        max_features=10000,
        use_idf=True,
        norm='l2',
        ngram_range=(1, 2)
    )

    ml_pipe = Pipeline(
        [('vect', tfidf_vect),
         ('clf', model(**model_params))]
    )

    ml_pipe.fit(training_df['cleaned_text'],
                 training_df['encoded_classification'])
    return ml_pipe

def read_training_set(
        proposal_data_dir="/Users/nmiles/PACMan_dist/proposal_data/",
        proposal_cycle='25',
        classifications='/Users/nmiles/PACMan_dist/cycle_25_classifications.txt'
):
    flist = glob.glob(
        f"{proposal_data_dir}/Cy{proposal_cycle}_Proposals_txt"
        f"/training_corpus/*training.txt"
    )
    proposal_numbers = [int(val.split('/')[-1].split('_')[0]) for val in flist]
    flist_and_pnum = list(zip(flist, proposal_numbers))
    flist_and_pnum.sort(key=lambda val: val[1])
    flist_sorted, proposal_num = list(zip(*flist_and_pnum))
    proposal_classifications = pd.read_csv(classifications)

    # Generate a new columnn to store the filenames and initialize it with NaNs
    proposal_classifications['fname'] = [np.nan] * len(
        proposal_classifications)

    # Loop through each proposal and update the dataframe with the filename
    for num, fname in zip(proposal_num, flist_sorted):
        proposal_classifications['fname'].loc[num - 1] = fname

    # Drop any rows that have nan
    final_df = proposal_classifications.dropna()
    encoder = LabelEncoder()
    encoder.fit(final_df['classification'])
    nl = '\n'
    print(f"The identified classes are:\n{nl.join(encoder.classes_)}")
    encoded_values = encoder.transform(final_df['classification'])
    final_df['encoded_classification'] = encoded_values
    text_df = read_in_dataset(flist=final_df['fname'].values)
    return text_df







if __name__ == '__main__':
    pass