#!/usr/bin/env python
import argparse
import glob
import logging
import os
import re

import tqdm

logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s.%(funcName)s:%(lineno)d]'
                           ' %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument(
    '-v', '--verbose',
    help="Be verbose", 
    default="NOTSET",
    action="store_const", 
    dest="loglevel",
     const=logging.INFO,
)
args = parser.parse_args()
LOG = logging.getLogger('proposal_scraper')
LOG.setLevel(level=args.loglevel)

class ProposalScraper(object):
    """
    A class for handling the scraping of the text files produced by the pdf to
    ascii converter.

    """

    def __init__(self):
        self._fname = None
        self._archival = False
        self._proposal_label = {
            'Scientific Category': None,
            'Scientific Keywords': None,
        }

        self._section_data = {
            'Abstract': None,
            'Investigators': None,
            'Scientific Justification': None,
            'Description of the Observations': None,
            'Special Requirements': None,
            'Justify Duplications': None,
            'Analysis Plan': None
        }

        self._section_data_archival = {
            'Abstract': None,
            'Investigators': None,
            'Scientific Justification': None,
            'Analysis Plan': None,
            'Management Plan': None
        }

        self._text = None

    @property
    def archival(self):
        """Flag to specify if proposal is AR or not"""
        return self._archival

    @archival.setter
    def archival(self, value):
        self._archival = value

    @property
    def fname(self):
        """File to process"""
        return self._fname

    @fname.setter
    def fname(self, value):
        self._fname = value

    @property
    def proposal_label(self):
        """Keywords to save from the proposal template"""
        return self._proposal_label

    @proposal_label.setter
    def proposal_label(self, value):
        self._proposal_label = value

    @property
    def section_data(self):
        """Section Names defined in the Phase I template"""
        return self._section_data

    @section_data.setter
    def section_data(self, value):
        self._section_data = value

    @property
    def section_data_archival(self):
        """Section Names defined in the Phase I template"""
        return self._section_data_archival

    @section_data_archival.setter
    def section_data_archival(self, value):
        self._section_data_archival = value

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = value

    def read_file(self, fname=None):
        """ Read the file and determine the proposal type

        Parameters
        ----------
        fname : str
            File to process

        Returns
        -------

        """
        # Create regex pattern for removing punctuation while keeping
        # remove = string.punctuation
        # remove = remove.replace("-", "") # don't remove hyphens
        # pattern = r"[{}]".format(remove) # create the pattern
        if fname is not None:
            self.fname = fname

        with open(self.fname, 'r') as fobj:
            lines = fobj.readlines()
            text = [line.strip('\n') for line in lines]
            text = [sentence for sentence in text if sentence]
            text = [
                re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', ' ', sentence)
                for sentence in text
            ]
            # text = [
            #     re.sub(r'[^a-zA-Z0-9-]+', ' ', sentence)
            #     for sentence in text
            # ]
        self._text = text

        if 'AR' in self.text[1]:
            self.archival = True
        else:
            self.archival = False

    def extract_keywords(self):
        """Extract the keyword info above the abstract section"""
        i=0
        while self.text[i] != 'Abstract':
            for key in self.proposal_label.keys():
                if key in self.text[i]:
                    self.proposal_label[key] = self.text[i].split(':')[-1]
            i += 1

    def extract_sections(self):
        
        # Set the line number counter
        current_line_num = 0
        
        # Check to see if the proposal is archival or not and grab the correct section names
        if self.archival:
            section_names = list(self.section_data_archival.keys())
            section_data = self.section_data_archival
        else:
            section_names = list(self.section_data.keys())
            section_data = self.section_data

        # Initialize the current idx for the section_names list
        current_section_idx = 0
        
        # Initialize a variable for computing the index of the next section.
        # This is required because proposals might be missing sections and so
        # when look for all data between two sections we need to have fine control
        # over what section is used to define the end of the current section
        next_idx = 1
        
        # Start looping through the text from line 0
        while current_line_num < len(self.text):
            # The current section we are searching for
            current_section = section_names[current_section_idx]
            
            # Line number corresponding to start and stop of current section
            section_start = None
            section_end = None
            
            # If we find our section name in the current line number, this begins the
            # start of our section. 
            if current_section in self.text[current_line_num]:
                # Section headers have their own line and so the true start is the next index
                section_start = current_line_num + 1 
                LOG.info(f'{current_section} starts on line {section_start}')
                
                # Compute the index for the next section in the list 
                next_section_idx = current_section_idx + next_idx
                try:
                    next_section = section_names[next_section_idx]
                except IndexError:
                    # If we've exhausted all section names, write the remaining portion of the file into the current section
                    LOG.info('Exhausted all section names, writing remaining lines into current section')
                    section_end = len(self.text)
    #                 print(f'Extracting lines {section_start} to {section_end} for {current_section}')
                    section_data[current_section] = self.text[section_start:section_end]
                    break
                
                # If we haven't exhausted all section names, continue looping through lines
                # until we've found the next section
    #             print(f'Looking for text between {current_section} and {next_section}')
                j = section_start
                while j < len(self.text):
                    text = self.text[j]
                    # if the next section title is in the current text, we've found the end
                    if next_section in text:
                        section_end = j - 1
                        LOG.info(f'{current_section} ends on line {section_end}')
                        current_section_idx +=1
                        # Set the current line number to the end of the section we just found
                        # This ensure the loops picks up on the next section
                        current_line_num = j - 1
                        break
                        
                    # Increase j by one to step to the next line
                    j += 1
                    
                    # If we hit the end of the file and we never found the section end
                    # increase the next_idx by one and search for the next section
                    if j >= len(self.text) and section_end is None :
                        LOG.info(f'Reached the end of file without finding {next_section}')
                        j = section_start
                        next_idx +=1
                        next_section_idx = current_section_idx + next_idx
                        try:
                            next_section = section_names[next_section_idx]
                        except IndexError as e:
                            # Exhausted the list again!
                            section_end = len(self.text)
                            current_line_num = section_end
                            break
                        else:
                            LOG.info(f'Restarting from line {section_start} and searching for text between {current_section} {section_names[next_section_idx]}')

                LOG.info(f'Extracting lines {section_start} to {section_end} for {current_section}')
                section_data[current_section] = self.text[section_start:section_end]

                current_section_idx = next_section_idx
                next_idx = 1
            current_line_num+=1
        if self.archival:
            self.section_data_archival = section_data
        else:
            self.section_data = section_data


    # def extract_sections(self):
    #     i = 0
    #     section_start = None
    #     section_end = None
    #     if self.archival:
    #         section_names = list(self.section_data_archival.keys())
    #         section_data = self.section_data_archival
    #     else:
    #         section_names = list(self.section_data.keys())
    #         section_data = self.section_data

    #     current_section_idx = 0
    #     while i < len(self.text):
    #         current_section = section_names[current_section_idx]
    #         if current_section in self.text[i]:
    #             section_start = i + 1
    #             LOG.info(f'{current_section} starts on line {section_start}')
    #             j = i
    #             while j < len(self.text):
    #                 try:
    #                     next_section = section_names[current_section_idx+1]
    #                 except IndexError:
    #                     LOG.info('Exhausted all section names...')
    #                     # If we've exhausted all section names, write the remaining portion of the file into the last section
    #                     section_end = len(self.text)
    #                     i = len(self.text)
    #                     break

    #                 try:
    #                     t = self.text[j]
    #                 except IndexError:
    #                     # Break the loop, no need to keep going if we've
    #                     # made it through the text
    #                     break

    #                 if next_section in t:
    #                     section_end = j - 1
    #                     LOG.info(f'{current_section} ends on line {section_end}')
    #                     current_section_idx +=1
    #                     i = j - 1
    #                     break
    #                 j+=1
    #             msg = (
    #                 f"Extracting lines {section_start} to "+
    #                  f"{section_end} for {current_section}\n{'-'*79}"
    #                 )
    #             LOG.info(msg)
    #             section_data[current_section] = self.text[section_start:section_end]
    #         i+=1
    #     if self.archival:
    #         self.section_data_archival = section_data
    #     else:
    #         self.section_data = section_data


    def write_training_data(self, training_sections):
        """Write out the training data we will use for text classification
        """
        outdir = os.path.join(os.path.dirname(self.fname), 'training_corpus')
        try:
            os.mkdir(outdir)
        except FileExistsError:
            pass

        fout = os.path.basename(self.fname.replace('.txtx','_training.txt'))
        fout = os.path.join(outdir, fout)

        data = []
        for section in training_sections:
            try:
                if self.archival:
                    text = self.section_data_archival[section]
                else:
                    text = self.section_data[section]
                if isinstance(text, list):
                    text = '\n'.join(text)
            except KeyError:
                LOG.info(f'Missing info for {section}')
            else:
                # print(section, data[:10])
                data.append(text)

        data = '\n'.join(data)
        with open(fout, mode='w') as fobj:
            fobj.write(data)


    def write_section_data(self, section_name):
        """ Write out the data for the section specified by section_name

        Parameters
        ----------
        section_name : str

        Returns
        -------

        """
        outdir = os.path.join(os.path.dirname(self.fname), 'training_corpus')

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
            data = self.proposal_label[section_name]

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
            LOG.info(f)
            self.read_file(fname=f)
            self.extract_sections()
            self.extract_keywords()
            self.write_section_data(
                section_name='Scientific Category'
            )
            self.write_training_data(
                training_sections=['Abstract', 'Scientific Justification']
            )


def main():
    # fname = '/Users/nmiles/PACMan_dist/C25/0948.txtx'
    prop = ProposalScraper()
    prop.extract_flist()

if __name__ == '__main__':
    main()