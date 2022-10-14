import re

def remove_special_words(text):
    """
    Small process designed for text that has ALREADY been processed (ex: "8" -> "huit"), but some special words might still be present (ex: "<noise>")
    """
    if not text: return "" # sometimes when the text is empty ait turns out to None in CSV
    #if PRINT_TIMING:
    #    tic = time.time()
    try:
        text = re.sub(r"<.*?>", "", text)
    except:
        print("PROBLEM WITH TEXT:", text, type(text))
        text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"' ", "'", text)
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
    text = re.sub(r" +", " ", text).strip()
    text = text.lower() # TCOF

    # if PRINT_TIMING:
    #     print("Removing special words took %.3f s (%s)" % (time.time() - tic, text))
    return text