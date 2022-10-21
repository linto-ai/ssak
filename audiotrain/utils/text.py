import re

_whitespace_re = re.compile(r'[^\S\r\n]+')

def remove_special_words(text,
    glue_apostrophe = True,
    extra = True,
    ):
    """
    Small process designed for text that has ALREADY been processed (ex: "8" -> "huit"), but some special words might still be present (ex: "<noise>")
    """
    # sometimes empty text could have been transformed to None (ex: in CSV)
    if not text: return ""

    try:
        text = re.sub(r"<.*?>", "", text)
    except:
        print("PROBLEM WITH TEXT:", text, type(text))
        text = re.sub(r"<.*?>", "", text)
    
    if glue_apostrophe:
        text = re.sub(r"'[^\S\r\n]+", "'", text)
    else:
        text = re.sub(r"'", "' ", text).strip()

    if extra:
        text = re.sub(r"\" ", " ", text)
        text = re.sub(r":", " ", text)
        text = re.sub(r"\+", " plus ", text)
        text = re.sub(r"1", " un ", text)
        text = re.sub(r"2", " deux ", text)
        text = re.sub(r"3", " trois ", text)
        text = re.sub(r"4", " quatre ", text)
        text = re.sub(r"5", " cinq ", text)
        text = re.sub(r"6", " six ", text)
        text = re.sub(r"7", " sept ", text)
        text = re.sub(r"8", " huit ", text)
        text = re.sub(r"9", " neuf ", text)

    text = re.sub(_whitespace_re, ' ', text).strip()
    #text = re.sub(r" +", " ", text).strip()

    text = text.lower() # TCOF
    return text