.. _pacman:

pacman2020
===============

**Author**: Nathan Miles

This package is designed to handle the classification of text data scraped from
HST proposals. It contains all of the functionality required to build an
end-to-end machine learning pipeline for the classification of HST Proposals into
one of the HST science categories. In short, we provide tools for

    #. scraping HST Proposals that have been converted from a PDF to a plain text
       file,

    #. text pre-processing (e.g. tokenization, lemmatization, etc..) using `spaCy`,

    #. training and testing on hand classified proposals,

    #. classification of unclassified proposals.

.. toctree::
   :maxdepth: 2 

   pacman2020
   utils



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
