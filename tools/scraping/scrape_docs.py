#!/usr/bin/env python3

"""scrap.py: Scrape documents (docx and odt files)."""

__author__      = "Jerome Louradour"
__copyright__   = "Copyright 2022, Linagora"

import docx
import zipfile
import xml.etree.ElementTree as ET
import re

def doc2text(filename):
    if filename.endswith("docx"):
        document = docx.Document(filename)
        lines = [paragraph.text for paragraph in document.paragraphs]
    elif filename.endswith(".odt"):
        myfile = zipfile.ZipFile(filename)
        listoffiles = myfile.infolist()
        for s in listoffiles:
            if s.orig_filename == 'content.xml':
                bh = myfile.read(s.orig_filename)
                bh = bh.decode("utf8")
                # Remove tags that could break the text in several parts
                for i in range(2):
                    bh = re.sub(r'<text:span text:style-name="[^"]*">([^<]*)</text:span>', r'\1', bh)
                bh = re.sub('<text:tab/>', '', bh)
                # Parse the xml
                root = ET.fromstring(bh)
                lines = root.itertext()
    text = '\n'.join([w.strip() for w in lines if len(w.split()) > 1])
    return text
                



if __name__ == "__main__":

    import os
    import argparse

    parser = argparse.ArgumentParser(description='Scrape documents in a folder (docx and odt files).')
    parser.add_argument('input', type=str, help='Input folder.')
    parser.add_argument('output', type=str, help='Output folder.')
    parser.add_argument('output_formatted', type=str, help='Output folder for formatted text (assuming French language).', nargs="?", default = None)
    args = parser.parse_args()

    dir_in = args.input
    dir_out1 = args.output
    dir_out2 = args.output_formatted
    if dir_out2:
        from linastt.utils.text import format_text_fr
        os.makedirs(dir_out2, exist_ok = True)
    os.makedirs(dir_out1, exist_ok = True)

    for root, dirs, files in os.walk(dir_in):

        for file in files:

            filename = os.path.join(root, file)
            filenametxt = os.path.splitext(os.path.basename(filename))[0].replace(" ","_") + ".txt"

            if not filename.endswith(".docx") and not filename.endswith(".odt"):
                print("Ignoring", filename)
                continue
            
            text = doc2text(filename)
            with open(os.path.join(dir_out1, filenametxt), "w") as f:
                f.write(text)
            
            if dir_out2:
                text = format_text_fr(text)
                with open(os.path.join(dir_out2, filenametxt), "w") as f:
                    f.write(text)
