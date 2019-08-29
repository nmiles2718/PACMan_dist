#!/usr/bin/env python
import glob
import logging
import os
import re

import tqdm

logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s.%(funcName)s:%(lineno)d]'
                           ' %(message)s')

LOG = logging.getLogger('proposal_scraper')
LOG.setLevel(logging.INFO)

class ProposalScraper(object):
    """
    A class for handling the scraping of the text files produced by the pdf to
    ascii converter.

    """

    def __init__(self, fname):
        self._fname = fname
        self._keywords = {
            'Scientific Category': None,
            'Scientific Keywords': None,
        }

        self._section_data = {
            'Abstract': None,
            'Scientific Justification': None,
            'Description of the Observations': None,
            'Special Requirements': None,
            'Justify Duplications': None,
            'Analysis Plan': None
        }

        self._text = None

    @property
    def fname(self):
        """File to process"""
        return self._fname

    @fname.setter
    def fname(self, value):
        self._fname = value

    @property
    def keywords(self):
        """Keywords to save from the proposal template"""
        return self._keywords

    @keywords.setter
    def keywords(self, value):
        self._keywords = value

    @property
    def section_data(self):
        """Section Names defined in the Phase I template"""
        return self._section_data

    @section_data.setter
    def section_data(self, value):
        self._section_data = value

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = value

    def read_file(self, fname=None):
        """ Read the file

        Parameters
        ----------
        fname : str
            File to process

        Returns
        -------

        """
        if fname is not None:
            self.fname = fname
        with open(self.fname, 'r') as fobj:
            lines = fobj.readlines()
            text = [line.strip('\n') for line in lines]
            text = [sent for sent in text if sent]
            text = [
                re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', sent)
                for sent in text
            ]

        self._text = text

    def extract_keywords(self):
        """Extract the keyword info above the abstract section"""
        i=0
        while self.text[i] != 'Abstract':
            for key in self.keywords.keys():
                if key in self.text[i]:
                    self.keywords[key] = self.text[i].split(':')[-1]
            i += 1

    def extract_sections(self):
        i = 0
        section_start = None
        section_end = None
        section_names = list(self.section_data.keys())
        while i < len(self.text):
            for k, key in enumerate(section_names):
                if k+1 == len(section_names):
                    last = True

                if key in self.text[i]:
                    section_start = i
                    j = i
                    while j < len(self.text):
                        try:
                            sname = section_names[k+1]
                        except IndexError:
                            # Write out everything below the last element to
                            # the Analysis Plan section
                            sname = section_names[k]

                        try:
                            t = self.text[j]
                        except IndexError:
                            # Break the loop, no need to keep going if we've
                            # made it through the test
                            break

                        if sname in t:
                            section_end = j
                            i = j
                            break

                        j+=1

                    self.section_data[key] = self.text[
                                             section_start:section_end
                                             ]
            i+=1

    def write_data(self, section_name):
        """ Write out the data for the section specified by section_name

        Parameters
        ----------
        section_name : str

        Returns
        -------

        """
        outdir = os.path.join(os.path.dirname(self.fname), 'sci_just')

        try:
            os.mkdir(outdir)
        except FileExistsError:
            pass

        fout = os.path.basename(self.fname).replace(
            ".txtx",
            f"_{section_name.replace(' ', '_')}.txt"
        )
        fout = os.path.join(outdir, fout)

        try:
            data = self.section_data[section_name]
        except KeyError:
            data = self.keywords[section_name]

        if data is None:
            LOG.info('\nOops! Nothing to write out.\n'
                     'You must execute the .extract_sections() method first.')
        else:
            if isinstance(data, list) :
                data = '\n'.join(data)

            with open(fout, mode='w') as fobj:
                fobj.write(data)
            # LOG.info(f'Successfully wrote  results to {fout}')

    def extract_flist(self, flist=None):
        """Extract the Sci. Jus. section for every file in flist

        Parameters
        ----------
        flist : list
            List of files to process

        Returns
        -------

        """
        if flist is None:
            flist = glob.glob('/Users/nmiles/PACMan_dist/C25/*txtx')

        for f in tqdm.tqdm(flist):
            self.read_file(fname=f)
            self.extract_sections()
            self.extract_keywords()
            self.write_data(section_name='Scientific Justification')


def main():
    fname = '/Users/nmiles/PACMan_dist/C25/0948.txtx'
    prop = ProposalScraper(fname=fname)
    prop.extract_flist()

if __name__ == '__main__':
    main()