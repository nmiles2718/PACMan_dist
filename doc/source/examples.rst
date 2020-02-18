********
Examples
********

This page contains a collection of examples to show the user how to perform
operations of interest.


Example 1: Saturation Mask
==========================
For a given FITS image, instantiate an instance of the
:py:class:`~analyze.analyze.fits_handler_crj.FitsHandlerCRJ` class
and then read in the data and generate a mask for saturated sources and
bad pixels.

Once the data has been read in and the mask has been generated, use the
:py:meth:`~analyze.analyze.Analyzer.show_stars` method from the parent
class and plot the data and mask for chip 1. When ``save=True`` is passed
to the :py:meth:`~analyze.analyze.Analyzer.show_stars` method, the
resulting plot will be saved to the `~/photCTE/plots` directory.

>>> from analyze import fits_handler_crj as fh_crj
>>> fits_obj = fh_crj.FitsHandlerCRJ('~/photCTE/data/14507/obs_set_13/jd4e03021_crj.fits')
>>> fits_obj.extract_data() # Read the SCI, ERR, DQ, WCS info and make a mask
>>> fits_obj.show_stars(data=fits_obj.chip1['sci'], mask=fits_obj.chip1['mask'], save=True) # Plot the data and save the output

 An example of the plot is shown below.

.. figure:: ./../../plots/example_mask.png
   :scale: 30 %
   :align: center


Example 2: Source Finding
=========================
For a given FITS image, instantiate an instance of the
:py:class:`~analyze.analyze.run_photometry_crj.Photometry` class and
plot the sources that are identified.

To accomplish this, we use the
:py:attr:`~analyze.analyze.Analyzer.results_dirs` attribute to find some
CRJs stored in one of the results directories for proposal 14507. Next, we use
the :py:class:`~analyze.analyze.fits_handler_crj.FitsHandlerCRJ` class to
read in the data and generate a mask for saturated sources and bad pixels.
Using the FILTER information for the given CRJ, we extract the filter-dependent
search parameters stored in the
:py:attr:`~analyze.analyze.Analyzer.phot_config` attribute and display them.
Finally, we run the source finding and generate a plot of the sources found.


>>> from analyze import fits_handler_crj as fh_crj
>>> from analyze import run_photometry_crj as phot_crj
>>> import glob
>>> p = phot_crj.Photometry('14507')
>>> crj_list = glob.glob(p.results_dirs[0]+'/*crj.fits')
>>> fits_obj = fh_crj.FitsHandlerCRJ(crj_list[0])
>>> fits_obj.extract_data()
>>> obs_filter = fits_obj.hdr['FILTER1']
>>> search_params = p.phot_config['FILTER'][obs_filter]
>>> print(search_params)
{'fwhm': 2.25, 'threshold': 10, 'sigma_radius': 1.25, 'sharphi': 1, 'sigma_clip_thresh': 3, 'sharplo': 0.5}
>>> sources = p.find_sources(fits_obj.chip1['sci'], fits_obj.chip1['mask'], **search_params)
>>> plot_params = {'ax_title': title, 'fout': 'example_sources.png', 'save': True, 'xlim': (1550, 1770), 'ylim': (1830, 1960)}
>>> fits_obj.show_stars(fits_obj.chip1['sci'], sources=sources, **plot_params)

An example of the plot produced is shown below.

.. figure:: ./../../plots/example_sources.png
   :scale: 75 %
   :align: center


Example 3: Comparing Catalogs
=============================
For a given FITS image, instatiate an instance of the
:py:class:`~analyze.analyze.compare_photometry.ComparePhotometry` class and
compare sources in each of the catalogs generated from the photometry step.

Once the instance has been created, we sort all of the catalogs in one of
the directories stored in the
:py:attr:`~analyze.analyze.Analyzer.results_dirs` attribute. This will return
a dictionary where the keys correspond to the image names and the values are
lists of two :py:attr:`~analyze.compare_photometry.ComparePhotometry.Data`
objects (one for each chip). Finally, we use the
:py:meth:`~analyze.compare_photometry.ComparePhotometry.compare_tables` method
to compare two catalogs and generate a plot with color-coded apertures to help
distinguish a given sources matched counterpart in the other image.

>>> from analyze import compare_photometry as comp_phot
>>> c = compare_phot.ComparePhotometry('14507')
>>> phot_data = c.sort_images(c.results_dirs[0])
>>> image_names = list(phot_data.keys())
>>> print(image_names)
['jd4e02041', 'jd4e02021']
>>> im1 = phot_data[image_names[0]][0] # chip 1 of first image
>>> im2 = phot_data[image_names[1]][1] # chip 2 of second image
>>> c.compare_tables(im1=im1, im2=im2, dirname=c.results_dirs[0], plot=True, save=True)

.. figure:: ./../../plots/example_matched_sources.png
   :scale: 75 %
   :align: center
