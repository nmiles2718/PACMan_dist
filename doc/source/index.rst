.. _pacman:

pacman2020
===============

**Author**: Nathan Miles

This package was written to facilitate the process of generating the data used
in computing the coefficients for the photometric CTE analysis. It is designed
to be operated as a single pipeline via the :py:mod:`pipeline` module. The
:py:mod:`pipeline` module has a command line interface designed to provide the
user with complete control over what steps are executed. The steps available
are listed below in the order they would be executed:

    #. Download observations for the specified proposal ID

    #. Sort the RAW files into grouped observation sets

    #. Process the sorted RAW with CALACS

       * Generates FLTs and CRJs, the CRJs will be normalized by their
         exposure time and multipled by the pixel area map (PAM)

    #. Sort the CRJs into the results directories based on the filter,
       exptime, and targname

       * Will rename the CRJs during the copying process to include text that
         describes which CTE analysis the file should be used for

    #. Drizzle the individual CRJ files to produce DRZ files

    #. Perform aperture photometry generating catalogs for each chip of each
       CRJ

    #. Compare the generated catalogs and find the common sources present in
       each pointing.

       * Create a master catalog with the necessary information
         for deriving the coefficients of the photometric CTE model

.. toctree::
   :maxdepth: 2 

   pacman2020



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
