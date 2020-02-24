.. _pacman:

pacman2020
===============

**Author**: Nathan Miles

This package is designed to handle the classification of text data scraped from
HST proposals. It contains all of the functionality required to build an
end-to-end machine learning pipeline for the classification of HST Proposals into
one of the HST science categories. In short, we provide tools for

    #. proposal scraping (:py:class:`~utils.proposal_scraper.HSTProposalScraper`),

    #. text preprocessing (:py:class:`~utils.tokenizer.PACManTokenizer`) using `spaCy <https://spacy.io>`_,

    #. training and testing on hand classified proposals (:py:class:`~pacman2020.PACManTrain`),

    #. classification of unclassified proposals (:py:class:`~pacman2020.PACManPipeline`).

.. toctree::
   :maxdepth: 2

   howto_training
   howto_classifying_new_data
   pacman2020
   utils


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
