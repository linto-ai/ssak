#!/usr/bin/env python3

"""scrap.py: Scrape documents (docx and odt files)."""

__author__      = "Jerome Louradour"
__copyright__   = "Copyright 2022, Linagora"

import os
import re
from enum import Enum
import warnings
import xml.etree.ElementTree as ET

class DocType(Enum):
    DOCX = 1
    ODT = 2
    PDF = 3
    XLS = 4

EXTENSIONS = {
    ".docx": DocType.DOCX,
    ".odt": DocType.ODT,
    ".pdf": DocType.PDF,
    ".xlsx": DocType.XLS,
    ".xlsm": DocType.XLS,
    ".xltx": DocType.XLS,
    ".xltm": DocType.XLS,
    ".ods": DocType.ODT,
}

def doc2text(filename):
    doctype = EXTENSIONS.get(os.path.splitext(filename)[1].lower())
    if doctype == DocType.DOCX:

        try:
            import docx
        except ImportError:
            raise RuntimeError("docx module not found. DOCX files will not be supported.")

        document = docx.Document(filename)
        lines = [paragraph.text for paragraph in document.paragraphs]
    
    elif doctype == DocType.ODT:

        try:
            import zipfile
        except ImportError:
            raise RuntimeError("zipfile module not found. ODT files will not be supported.")

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

    elif doctype == DocType.PDF:

        try:
            import fitz
        except ImportError:
            raise RuntimeError("fitz module not found. PDF files will not be supported.")

        global pdf_headers, pdf_footers
        pdf_headers, pdf_footers = [], []
        doc = fitz.open(filename)
        pages = list(doc)
        pages = [page.get_text(sort=True) for page in pages]
        pages = ["\n".join(extract_paragraph_in_pdf(page)) for page in pages]
        # reader = PdfReader(filename)
        # pages = [page.extract_text() for page in reader.pages]
        page = "\n".join(pages)
        lines = extract_paragraph_in_pdf(page)

    elif doctype == DocType.XLS:

        try:
            import openpyxl
        except ImportError:
            raise RuntimeError("openpyxl module not found. XLS files will not be supported.")

        wb = openpyxl.load_workbook(filename)
        ws = wb.active
        lines = []
        for row in ws.rows:
            for cell in row:
                if isinstance(cell.value, str):
                    lines += cell.value.split("\n")
    else:
        raise NotImplementedError(f"Extension {os.path.splitext(filename)[1]} not supported.")

    # Remove extra spaces and empty lines
    lines = [l.strip() for l in lines if len(l.strip()) > 0]
    
    # Remove duplicates and keep the order
    lines = list(dict.fromkeys(lines))

    text = '\n'.join(lines) 
    return text

pdf_headers = []
pdf_footers = []
def extract_paragraph_in_pdf(page):
    # Remove possible header
    global pdf_headers
    for header in pdf_headers:
        page = re.sub(rf"^\s*{re.escape(header)}\s*\n","", page)
    pdf_headers.append(page.split("\n")[0].strip())
    # Remove page number
    page = re.sub(r"\n\s*[0-9]+\s*$","\n", page)
    # Remove possible pdf_footers
    global pdf_footers
    for footer in pdf_footers:
        page = re.sub(rf"\s*{re.escape(footer)}\s*$","", page)
    pdf_footers.append(page.split("\n")[-1].strip())
    # Split into probable paragraphs
    lines = re.split(r'([\.?!»"])\s*\n', page)
    lines = ["".join([a,b]) for a,b in zip(lines[::2],lines[1::2]+[""])]
    # Remove double spaces
    lines = [re.sub(r"\s+"," ", line) for line in lines]
    # Remove repeated dots
    lines = [re.sub(r"(\.\s*){3,}\.",".", line) for line in lines]
    return lines


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

    assert os.path.isdir(dir_in), f"Input folder {dir_in} does not exist."

    if dir_out2:
        from linastt.utils.text import format_text_latin
        os.makedirs(dir_out2, exist_ok = True)
    os.makedirs(dir_out1, exist_ok = True)

    for root, dirs, files in os.walk(dir_in):

        for file in files:

            filename = os.path.join(root, file)
            if not os.path.splitext(filename)[1].lower() in EXTENSIONS:
                print("Ignoring", filename)
                continue

            b, e = os.path.splitext(os.path.basename(filename))
            if e:
                b+= "_"+e[1:]
            filenametxt = b.replace(" ","_") + ".txt"

            print("Processing", filename, "->", os.path.join(dir_out1, filenametxt))
            
            text = doc2text(filename)
            with open(os.path.join(dir_out1, filenametxt), "w") as f:
                f.write(text)
            
            if dir_out2:
                text = format_text_latin(text)
                with open(os.path.join(dir_out2, filenametxt), "w") as f:
                    f.write(text)
