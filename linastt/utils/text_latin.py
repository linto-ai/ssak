import re
import math
import warnings

from linastt.utils.text_utils import (
    collapse_whitespace,
    remove_special_characters,
    format_special_characters,
    regex_escape,
    transliterate,
    undigit,
    cardinal_numbers_to_letters,
    convert_symbols_to_words,
    _currencies
)

def _rm_key(d, key):
    d = d.copy()
    d.pop(key)
    return d

def find_acronyms(text, ignore_first_upper_words = True):
    if not text: return []
    i = 0
    if ignore_first_upper_words:
        # All the first upper case letters will be ignored
        up = text.upper()
        for j, (a, b) in enumerate(zip(text, up)):
            if a == " ":
                i = j
            if a != b:
                break
    return re.findall(r"\b[A-Z][A-Z0-9]{1,}\b", text[i:])

_ALL_ACRONYMS = []

def format_text_latin(text,
    lang = "fr",
    lower_case = True,
    keep_punc = False,
    remove_ligatures = True,
    extract_parenthesis = False,
    fid_acronyms = None,
    fid_special_chars = None,
    safety_checks = True,
    remove_suspicious_entry = False,
    ):

    opts = _rm_key(locals(), "text")

    text_orig = text

    try:

        # Recursive call (list)
        if isinstance(text, list):
            return [format_text_latin(t, **opts) for t in text]

        # Recursive call (line breaks)
        if "\n" in text:
            return "\n".join([format_text_latin(t, **opts) for t in text.split("\n")])
        
        # Recursive call (parenthesis)
        if extract_parenthesis and "(" in text and ")" in text:
            in_parenthesis = re.findall(r"\(([^\(\)]*?)\)", text)
            if len(in_parenthesis):
                in_parenthesis = [s.rstrip(")").lstrip("(") for s in in_parenthesis]
                regex = "("+")|(".join(["\("+regex_escape(p)+"\)" for p in in_parenthesis])+")"
                without_parenthesis = re.sub(regex, "", text)
                # assert without_parenthesis != text
                if without_parenthesis != text: # Avoid infinite recursion
                    texts = [without_parenthesis] + in_parenthesis
                    return "\n".join([format_text_latin(t, **opts) for t in texts])

        if remove_suspicious_entry:
            # Assuming no letter can be repeated 3 times in French
            if re.findall(r"([a-z])\1{2,}", text):
                return ""
            # Assuming no words start with double letters in French
            if re.findall(re.compile(r"\b([a-z])\1{1,}", re.IGNORECASE), transliterate(text)):
                return ""
            if "familyfont" in text:
                return ""

        global _ALL_ACRONYMS

        # Reorder currencies (1,20€ -> 1 € 20)
        coma = "," if lang in ["fr"] else "\."
        for c in _currencies:
            if c in text:
                c = regex_escape(c)
                text = re.sub(r"\b(\d+)" + coma + r"(\d+)\s*" +
                            c, r"\1 " + c + r" \2", text)

        # Roman digits
        if re.search(r"[IVX]", text):
            if lang == "en":
                digits = re.findall(
                    r"\b(?=[XVI])M*(XX{0,3})(I[XV]|V?I{0,3})(º|st|nd|rd|th)?\b", text)
                digits = ["".join(d) for d in digits]
            elif lang == "fr":
                digits = re.findall(
                    r"\b(?=[XVI])M*(XX{0,3})(I[XV]|V?I{0,3})(º|ème|eme|e|er|ère)?\b", text)
                digits = ["".join(d) for d in digits]
            else:
                digits = re.findall(
                    r"\b(?=[XVI])M*(XX{0,3})(I[XV]|V?I{0,3})\b", text)
                digits = ["".join(d) for d in digits]
            if digits:
                digits = sorted(list(set(digits)), reverse=True,
                                key=lambda x: (len(x), x))
                for s in digits:
                    filtered = re.sub("[a-zèº]", "", s)
                    ordinal = filtered != s
                    digit = roman_to_decimal(filtered)
                    v = undigit(str(digit), lang=lang,
                                to="ordinal" if ordinal else "cardinal")
                    text = re.sub(r"\b" + s + r"\b", v, text)
        
        if fid_acronyms is not None:
            acronyms = find_acronyms(text)
            for acronym in acronyms:
                if acronym not in _ALL_ACRONYMS:
                    print(acronym, file = fid_acronyms)
                    fid_acronyms.flush()
                    _ALL_ACRONYMS.append(acronym)

        if lower_case:
            text = text.lower()
            if remove_ligatures:
                text = re.sub(r"œ", "oe", text)
                text = re.sub(r"æ", "ae", text)
                text = re.sub(r"ﬁ", "fi", text)
                text = re.sub(r"ﬂ", "fl", text)
                text = re.sub("ĳ", "ij", text)
        elif remove_ligatures:
            text = re.sub(r"œ", "oe", text)
            text = re.sub(r"æ", "ae", text)
            text = re.sub(r"ﬁ", "fi", text)
            text = re.sub(r"ﬂ", "fl", text)
            text = re.sub("ĳ", "ij", text)
            text = re.sub(r"Œ", "OE", text)
            text = re.sub(r"Æ", "AE", text)

        text = re.sub("``", "\"", text)
        text = re.sub("''", "\"", text)
        text = re.sub("-+", "-", text) # ---- -> -

        for reg, replacement in _corrections_caracteres_speciaux_fr:
            text = re.sub(reg, replacement, text)

        text = ' '+text+' '

        numbers=re.findall(r"\d{1,3}(?:[\.,]000)+",text)
        for n in numbers:
            text = re.sub(n,re.sub(r"[,.]","",n), text)

        # Replace "." by "point" and "/" by "slash" in internet websites
        # Find all the websites in the text
        websites = [w for w in re.findall('(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-&?=%.]+', text) if ".." not in w]
        websites = sorted(set(websites), key = len, reverse = True)
        for w in websites:
            w2 = w
            w2 = re.sub("\.", " point ", w2)
            w2 = re.sub(":", " deux points ", w2)
            w2 = re.sub("/", " slash ", w2)
            w2 = re.sub("-", " tiret ", w2)
            #text = re.sub(w, w2, text)
            text = text.replace(w, w2)

        # Abbréviations
        text = re.sub(" m\. "," monsieur ",text)
        text = re.sub(" mme\.? ", " madame ",text)
        text = re.sub(" mlle\.? ", " mademoiselle ",text)

        text = re.sub(r"[’‘]","'", text)
        text = re.sub("'","' ", text)
        text = re.sub('"',' " ', text)
        text = re.sub("' '", "''", text)
        text = re.sub(":", " : ", text)
        text = re.sub(";", " ; ", text)
        text = re.sub(',|¸',',', text)
        text = re.sub(", ", " , ", text)
        text = re.sub("\!", " ! ", text)
        text = re.sub("\?", " ? ", text)
        #text = re.sub("^ *-+", "", text)
        text = re.sub("\^+","", text)
        text = re.sub(" +(- +)+", " ", text)
        text = re.sub("- ", " ", text)
        #text = re.sub("([a-zàâäçèéêëîïôùûü]+)- +", r"\1-", text)
        #text = re.sub(" -([a-zàâäçèéêëîïôùûü]+)", r"-\1", text)
        text = re.sub("([,;:\!\?\.]) -([a-zàâäçèéêëîïôùûü]+)", r"\1 \2", text)
        text = re.sub("([a-zàâäçèéêëîïôùûü]{3,})' ", r"\1 ", text)
        text = re.sub("([a-zàâäçèéêëîïôùûü]{2,})' *[,;:\!\?\.]", r"\1 ", text)
        text = re.sub('\.{2,}',' ', text)
        text = re.sub('\. *$',' . ', text)
        text = re.sub('(\d)\. ',r'\1 . ', text)

        text=re.sub('\{',' { ',text)
        text=re.sub('\}',' } ',text)
        text=re.sub('\(',' ( ',text)
        text=re.sub('\)',' ) ',text)
        text=re.sub('\[',' [ ',text)
        text=re.sub('\]',' ] ',text)
        text=re.sub(r"<([^<>]*)>",r"\1",text)

        for reg, replacement in _corrections_regex_fr:
            text = re.sub(reg, replacement, text)

        heures=re.findall("\d+ *h *\d+",text)
        for h in heures:
            split_h=h.split('h')
            text_rep=re.sub('^0+','',split_h[0])+' heures '+re.sub('^0+','',split_h[1])
            #text_rep=split_h[0]+' heures '+split_h[1]
            text=text.replace(h, text_rep)

        text = re.sub("(\d+)''",r"\1 secondes ",text)
        text = re.sub("(\d+)'",r"\1 minutes ",text)
        #text = re.sub("(\d+)°",r"\1 degrés ",text)

        # Ordinal digits
        if lang == "en":
            digits = re.findall(
                r"\b\d*1(?:st)|\d*2(?:nd)|\d*3(?:rd)|\d+(?:º|th)\b", text)
        elif lang == "fr":
            digits = re.findall(
                r"\b1(?:ère|ere|er|re|r)|2(?:nd|nde)|\d+(?:º|ème|eme|e)\b", text)
        else:
            warnings.warn(
                f"Language {lang} not supported for some normalization. Some words might be mis-localized.")
            digits = []
        if digits:
            digits = sorted(list(set(digits)), reverse=True,
                            key=lambda x: (len(x), x))
            for digit in digits:
                word = undigit(re.findall(r"\d+", digit)
                            [0], to="ordinal", lang=lang)
                text = re.sub(r'\b'+str(digit)+r'\b', word, text)

        # Cardinal digits
        text = cardinal_numbers_to_letters(text, lang=lang)

        # Symbols (currencies, percent...)
        text = convert_symbols_to_words(text, lang, lower_case=lower_case)

        if safety_checks:
            if re.findall(r"\d", text):
                raise ValueError(f"Failed to convert all digits to words\nInput: {text_orig}\nOutput: {text}")

        # Dashes
        text = re.sub(r"((?:^)|(?:\W))[-_]",r"\1, ", text) # " j'ai dit -à ma ``belle-mère''-casse-toi" -> " j'ai dit , à ma ``belle-mère'', casse-toi"
        # Find all words with at least 2 dashes
        for word in sorted(list(set(re.findall(r"\b[a-z]+(?:-[a-z]+){2,}\b", text))), key = len, reverse = True):
            if "http" not in word and "www" not in word and not (len(re.findall("-", word)) in [2, 3] and word.split("-")[-2].lower() in ["de", "du", "des", "sur", "sous", "en", "au", "à", "le", "la", "les", "lès", "saint", "sainte", "grand", "t", "vous", "el", "al"]):
                subwords = word.split("-")
                if not (len(subwords) in [2, 3] and subwords[-2].lower() in ["de", "du", "des", "sur", "sous", "en", "au", "à", "le", "la", "les", "lès", "saint", "sainte", "grand", "t", "vous", "el", "al"]) \
                    and not min([(w in _all_nums) or (w.endswith("s") and w[:-1] in _all_nums) for w in subwords]):
                        # Replace all dashes by spaces
                        text = re.sub(r"\b"+word+r"\b", re.sub("-", " ", word), text)

        for reg, replacement in _corrections_abbreviations_fr:
            text = re.sub(reg, replacement, text)

        for reg, replacement in _multi_spelling_words:
            text = re.sub(reg, replacement, text)

        # Special Symbols
        text = format_special_characters(text)

        if not keep_punc:
            text = re.sub(r',|;|:|\!|\?|/|\.',' ',text)

        text = remove_special_characters(text, replace_by = "", latin_characters_only = True, fid = fid_special_chars)

        # # Non printable characters
        # if '\x81' in text:
        #     #i = text.index('\x81')
        #     #print("WARNING: weird character in text: ", text[:i], "\\x81", text[i+1:])
        #     text = text.replace('\x81', ' ')

        text = collapse_whitespace(text)

    except Exception as e:
        print(f"ERROR with text: {text_orig}")
        raise e

    return text

def roman_to_decimal(str):
    def value(r):
        if (r == 'I'):
            return 1
        if (r == 'V'):
            return 5
        if (r == 'X'):
            return 10
        if (r == 'L'):
            return 50
        if (r == 'C'):
            return 100
        if (r == 'D'):
            return 500
        if (r == 'M'):
            return 1000
        return -1

    res = 0
    i = 0
    while (i < len(str)):
        s1 = value(str[i])
        if (i + 1 < len(str)):
            s2 = value(str[i + 1])
            if (s1 >= s2):
                # Value of current symbol is greater or equal to the next symbol
                res = res + s1
                i = i + 1
            else:
                # Value of current symbol is greater or equal to the next symbol
                res = res + s2 - s1
                i = i + 2
        else:
            res = res + s1
            i = i + 1
    return res

#sorted(list(set([item for sublist in [w.split() for w in [num2words(i, lang='fr') for i in list(range(17)) + [i*10 for i in range(1,11)] + [1000**i for i in range(1,202)]]] for item in sublist])),key = len)
_all_nums = [
 'un',
 'dix',
 'six',
 'sept',
 'onze',
 'cinq',
 'huit',
 'cent',
 'neuf',
 'deux',
 'zéro',
 'mille',
 'trois',
 'seize',
 'vingt',
 'douze',
 'treize',
 'quatre',
 'trente',
 'quinze',
 'million',
 'billion',
 'trillion',
 'quarante',
 'soixante',
 'quatorze',
 'milliard',
 'billiard',
 'decillion',
 'cinquante',
 'nonillion',
 'trilliard',
 'octillion',
 'nonilliard',
 'decilliard',
 'sextillion',
 'octilliard',
 'septillion',
 'centillion',
 'quintillion',
 'sextilliard',
 'quadrillion',
 'septilliard',
 'centilliard',
 'undecillion',
 'undecilliard',
 'quintilliard',
 'tredecillion',
 'sexdecillion',
 'soixante-dix',
 'quadrilliard',
 'duodecillion',
 'vigintillion',
 'quindecillion',
 'trigintillion',
 'tredecilliard',
 'octodecillion',
 'vigintilliard',
 'duodecilliard',
 'sexdecilliard',
 'quatre-vingts',
 'septdecillion',
 'sexagintillion',
 'octogintillion',
 'octodecilliard',
 'novemdecillion',
 'trigintilliard',
 'nonagintillion',
 'septdecilliard',
 'unvigintillion',
 'quindecilliard',
 'unvigintilliard',
 'sexagintilliard',
 'octogintilliard',
 'duovigintillion',
 'sexvigintillion',
 'trevigintillion',
 'novemdecilliard',
 'untrigintillion',
 'nonagintilliard',
 'septuagintillion',
 'trevigintilliard',
 'unoctogintillion',
 'septvigintillion',
 'tretrigintillion',
 'duovigintilliard',
 'untrigintilliard',
 'octovigintillion',
 'quatre-vingt-dix',
 'quinvigintillion',
 'duotrigintillion',
 'unsexagintillion',
 'sexvigintilliard',
 'unnonagintillion',
 'quadragintillion',
 'sextrigintillion',
 'quintrigintillion',
 'novemvigintillion',
 'quinvigintilliard',
 'treoctogintillion',
 'sexnonagintillion',
 'sextrigintilliard',
 'trenonagintillion',
 'octotrigintillion',
 'septuagintilliard',
 'quadragintilliard',
 'septtrigintillion',
 'tretrigintilliard',
 'duotrigintilliard',
 'unoctogintilliard',
 'duooctogintillion',
 'unsexagintilliard',
 'duononagintillion',
 'sexsexagintillion',
 'octovigintilliard',
 'tresexagintillion',
 'duosexagintillion',
 'sexoctogintillion',
 'unnonagintilliard',
 'quattuordecillion',
 'quinquagintillion',
 'septvigintilliard',
 'quinnonagintillion',
 'quinoctogintillion',
 'tresexagintilliard',
 'quattuordecilliard',
 'octosexagintillion',
 'unquadragintillion',
 'septtrigintilliard',
 'septsexagintillion',
 'novemtrigintillion',
 'octooctogintillion',
 'duooctogintilliard',
 'novemvigintilliard',
 'treoctogintilliard',
 'octotrigintilliard',
 'quinsexagintillion',
 'sexnonagintilliard',
 'septoctogintillion',
 'sexoctogintilliard',
 'septnonagintillion',
 'octononagintillion',
 'sexsexagintilliard',
 'duononagintilliard',
 'duosexagintilliard',
 'quintrigintilliard',
 'trenonagintilliard',
 'quinquagintilliard',
 'unseptuagintillion',
 'quinsexagintilliard',
 'unseptuagintilliard',
 'sexquadragintillion',
 'septoctogintilliard',
 'treseptuagintillion',
 'octooctogintilliard',
 'novemnonagintillion',
 'novemoctogintillion',
 'septnonagintilliard',
 'trequadragintillion',
 'octosexagintilliard',
 'quinoctogintilliard',
 'quinnonagintilliard',
 'duoquadragintillion',
 'unquinquagintillion',
 'novemtrigintilliard',
 'sexseptuagintillion',
 'duoseptuagintillion',
 'unquadragintilliard',
 'octononagintilliard',
 'novemsexagintillion',
 'septsexagintilliard',
 'octoseptuagintillion',
 'sexquadragintilliard',
 'novemoctogintilliard',
 'treseptuagintilliard',
 'duoquadragintilliard',
 'duoseptuagintilliard',
 'septquadragintillion',
 'quinquadragintillion',
 'duoquinquagintillion',
 'sexseptuagintilliard',
 'trequinquagintillion',
 'sexquinquagintillion',
 'unquinquagintilliard',
 'trequadragintilliard',
 'novemsexagintilliard',
 'novemnonagintilliard',
 'octoquadragintillion',
 'septseptuagintillion',
 'quinseptuagintillion',
 'quattuorvigintillion',
 'octoquinquagintillion',
 'quattuortrigintillion',
 'septquadragintilliard',
 'octoquadragintilliard',
 'quinseptuagintilliard',
 'septseptuagintilliard',
 'quattuorvigintilliard',
 'duoquinquagintilliard',
 'octoseptuagintilliard',
 'trequinquagintilliard',
 'novemseptuagintillion',
 'quinquinquagintillion',
 'novemquadragintillion',
 'septquinquagintillion',
 'quinquadragintilliard',
 'sexquinquagintilliard',
 'novemquadragintilliard',
 'octoquinquagintilliard',
 'septquinquagintilliard',
 'quattuorsexagintillion',
 'novemquinquagintillion',
 'quinquinquagintilliard',
 'quattuornonagintillion',
 'quattuoroctogintillion',
 'quattuortrigintilliard',
 'novemseptuagintilliard',
 'quattuorsexagintilliard',
 'novemquinquagintilliard',
 'quattuoroctogintilliard',
 'quattuornonagintilliard',
 'quattuorseptuagintillion',
 'quattuorquadragintillion',
 'quattuorquinquagintillion',
 'quattuorseptuagintilliard',
 'quattuorquadragintilliard',
 'quattuorquinquagintilliard',
]



_corrections_abbreviations_fr = [(r' '+x[0]+r' ', ' '+x[1]+' ') for x in [
    ("g", "grammes"),
    ("µg", "microgrammes"),
    ("μg", "microgrammes"),
    ("mg", "milligrammes"),
    ("kg", "kilogrammes"),
    ("mm", "millimètres"),
    ("cm", "centimètres"),
    # ("l", "litres"), # Caution with "l'"
    ("ml", "millilitres"),
    ("cm2", "centimètres carrés"),
]] + [
    ("@", " arobase "),
]


_corrections_caracteres_speciaux_fr = [(re.compile('%s' % x[0]), '%s' % x[1])
                  for x in [
                    (" ", " "),
                    ("а","a"),
                    ("â","â"),
                    ("à","à"),
                    ("á","á"),
                    ("ã","à"),
                    ("ā","a"),
                    ("ă","a"),
                    ("ǎ","a"),
                    ("е","e"),
                    ("ê","ê"),
                    ("é","é"),
                    ("è","è"),
                    ("ē","e"),
                    ("ĕ","e"),
                    ("ė","e"),
                    ("ę","e"),
                    ("ě","e"),
                    ("ё","e"),
                    ("ϊ","ï"),
                    ("ΐ","ï"),
                    ("ĩ","i"),
                    ("ī","i"),
                    ("ĭ","i"),
                    ("į","i"),
                    ("į","i"),
                    ("î","î"),
                    ("ı","i"),
                    ("ô","ô"),
                    ("ό","ο"),
                    ("ǒ","o"),
                    ("ō","o"),
                    ("ő","o"),
                    ("û","û"),
                    ("ǔ","u"),
                    ("ǜ","ü"),
                    ("ύ","u"),
                    ("ū","u"),
                    ("ŷ","y"),
                    ("ć","c"),
                    ("č","c"),
                    ("ƒ","f"),
                    ("ĝ","g"),
                    ("ğ","g"),
                    ("ġ","g"),
                    ("ĥ","h"),
                    ("ķ","k"),
                    ("ł","l"),
                    ("ń","n"),
                    ("ņ","n"),
                    ("ň","n"),
                    ("ř","r"),
                    ("ś","s"),
                    ("ş","s"),
                    ("š","s"),
                    ("ș","s"),
                    ("ţ","t"),
                    ("ț","t"),
                    ("ť","t"),
                    ("ŵ","w"),
                    ("ź","z"),
                    ("ż","z"),
                    ("ž","z"),
                    ("Ã","a"),
                    # ('À','À'),
                    # ('É','É'),
                    # ('È','È'),
                    # ('Â','Â'),
                    # ('Ê','Ê'),
                    # ('Ç','Ç'),
                    # ('Ù','Ù'),
                    # ('Û','Û'),
                    # ('Î','Î'),
                    ("×", " fois "),
                    ("÷", " divisé par "),
                    ('ａ', 'a'), ('ｂ', 'b'), ('ｃ', 'c'), ('ｄ', 'd'), ('ｅ', 'e'), ('ｆ', 'f'), ('ｇ', 'g'), ('ｈ', 'h'), ('ｉ', 'i'), ('ｊ', 'j'), ('ｋ', 'k'), ('ｌ', 'l'), ('ｍ', 'm'), ('ｎ', 'n'), ('ｏ', 'o'), ('ｐ', 'p'), ('ｑ', 'q'), ('ｒ', 'r'), ('ｓ', 's'), ('ｔ', 't'), ('ｕ', 'u'), ('ｖ', 'v'), ('ｗ', 'w'), ('ｘ', 'x'), ('ｙ', 'y'), ('ｚ', 'z'),
                    ("α", " alpha "),
                    ("β", " beta "),
                    ("γ", " gamma "),
                    ("δ", " delta "),
                    ("ε", " epsilon "),
                    ("ζ", " zeta "),
                    ("η", " eta "),
                    ("θ", " theta "),
                    ("ι", " iota "),
                    ("κ", " kappa "),
                    ("λ", " lambda "),
                    ("ν", " nu "),
                    ("ξ", " xi "),
                    ("ο", " omicron "),
                    ("π", " pi "),
                    ("ρ", " rho "),
                    ("σ", " sigma "),
                    ("τ", " tau "),
                    ("υ", " upsilon "),
                    ("φ", " phi "),
                    ("χ", " chi "),
                    ("ψ", " psi "),
                    ("ω", " omega "),
                    ("Α", " alpha "),
                    ("Β", " beta "),
                    ("Γ", " gamma "),
                    ("Δ", " delta "),
                    ("Ε", " epsilon "),
                    ("Ζ", " zeta "),
                    ("Η", " eta "),
                    ("Θ", " theta "),
                    ("Ι", " iota "),
                    ("Κ", " kappa "),
                    ("Λ", " lambda "),
                    ("Μ", " micro "),
                    ("Ν", " nu "),
                    ("Ξ", " xi "),
                    ("Ο", " omicron "),
                    ("Π", " pi "),
                    ("Ρ", " rho "),
                    ("Σ", " sigma "),
                    ("Τ", " tau "),
                    ("Υ", " upsilon "),
                    ("Φ", " phi "),
                    ("Χ", " chi "),
                    ("Ψ", " psi "),
                    ("Ω", " omega "),
                    ("♠", " pique "),
                    ("♣", " trèfle "),
                    ("♥", " coeur "),
                    ("♦", " carreau "),
                    ("♜", " tour "),
                    ("♞", " cavalier "),
                    ("♝", " fou "),
                    ("♛", " reine "),
                    ("♚", " roi "),
                    ("♟", " pion "),
                    ("♔", " roi "),
                    ("♕", " reine "),
                    ("♖", " tour "),
                    ("♗", " fou "),
                    ("♘", " cavalier "),
                    ("♙", " pion "),
                    ("♭", " bémol "),
                    ("♮", " dièse "),
                    ("♂", " mâle "),
                    ("♀", " femelle "),
                    ("☿", " mercure "),
                    ("∈", " appartient à "),
                    ("∉", " n'appartient pas à "),
                    ("∅", " vide "),
                    ("∪", " union "),
                    ("∩", " intersection "),
                    ("∧", " et "),
                    ("∨", " ou "),
                    ("∀", " pour tout "),
                    ("∃", " il existe "),
                    ("∂", " dérivée de "),
                    ("∇", " gradient de "),
                    ("√", " racine carrée de "),
                    ("∫", " intégrale de "),
                    ("∬", " double intégrale de "),
                    ("∭", " triple intégrale de "),
                    ("∮", " intégrale de surface de "),
                    ("∯", " double intégrale de surface de "),
                    ("∰", " triple intégrale de surface de "),
                    ("∴", " donc "),
                    ("∵", " car "),
                    ("∼", " environ "),
                    ("≈", " estime "),
                    ("≠", " différent de "),
                    ("≡", " égal à "),
                    ("≤", " inférieur ou égal à "),
                    ("≥", " supérieur ou égal à "),
                    ("⊂", " est inclus dans "),
                    ("⊃", " contient "),
                    ("⊄", " n'est pas inclus dans "),
                    ("⊆", " est inclus dans ou égal à "),
                    ("⊇", " contient ou est égal à "),
                    ("⊕", " addition "),
                    ("⊗", " multiplication "),
                    ("⊥", " perpendiculaire à "),
                    ("∑", " somme de "),
                    ("∏", " produit de "),
                    ("∐", " somme directe de "),
                    ("⇒", " implique "),
                    ("⇔", " équivaut à "),
                    ("⇐", " est impliqué par "),
                    ("⇆", " est équivalent à "),
                    ("⇎", " est défini par "),
                    ("ℤ", " entiers "),
                    ("ℚ", " rationnels "),
                    ("ℝ", " réels "),
                    ("ℂ", " complexes "),
                    ("ℕ", " naturels "),
                    ("ℵ", " aleph "),
                    ("ℶ", " beth "),
                    ("ℷ", " gimel "),
                    ("ℸ", " daleth "),
                    ("ℹ", " information "),
                ]]

_corrections_regex_fr = [(re.compile(' %s ' % x[0], re.IGNORECASE), ' %s ' % x[1])
                  for x in [
                    ("nº","numéro"),
                    ("n°","numéro"),
                    ("jus +qu'","jusqu'"),
                    ("pres +qu'","presqu'"),
                    ("lors +qu'","lorsqu'"),
                    ("quel +qu'","quelqu'"),
                    ("puis +qu'","puisqu'"),
                    ("aujour +d'","aujourd'"),
                    ("jusqu","jusqu'"),
                    ("presqu","presqu'"),
                    ("lorsqu","lorsqu'"),
                    ("quelqu","quelqu'"),
                    ("puisqu","puisqu'"),
                    ("aujourd","aujourd'"),
                    ("aujourd' +hui","aujourd'hui"),
                    ("quoiqu","quoiqu'"),
                    ("°", " degrés "),
                ]]

_multi_spelling_words = [(r'\b%s\b' % x[0], '%s' % x[1])
                  for x in [
                    ("ailloli", "aïoli"),
                    ("aillolis", "aïolis"),
                    ("aulne", "aune"),
                    ("aulnes", "aunes"),
                    ("bâiller", "bayer"),
                    ("bagout", "bagou"),
                    ("balluchon", "baluchon"),
                    ("balluchons", "baluchons"),
                    ("becqueter", "béqueter"),
                    ("bistrot", "bistro"),
                    ("bistrots", "bistros"),
                    ("bonbonne", "bombonne"),
                    ("bonbonnes", "bombonnes"),
                    ("cacahouète", "cacahuète"),
                    ("cacahouètes", "cacahuètes"),
                    ("cannette", "canette"),
                    ("cannettes", "canettes"),
                    ("caryatide", "cariatide"),
                    ("caryatides", "cariatides"),
                    ("chausse-trape", "chausse-trappe"),
                    ("chausse-trapes", "chausse-trappes"),
                    ("clef", "clé"),
                    ("clefs", "clés"),
                    ("cuiller", "cuillère"),
                    ("cuillers", "cuillères"),
                    ("démarcage", "démarquage"),
                    ("égrener", "égrainer"),
                    ("enraiement", "enraiement"),
                    ("etc", "et cetera"),
                    ("caetera", "cetera"),
                    ("cætera", "cetera"),
                    ("feignant", "fainéant"),
                    ("feignants", "fainéants"),
                    ("gri-gri", "grigri"),
                    ("gri-gris", "grigris"),
                    ("gris-gris", "grigris"),
                    ("hawaiien", "hawaïen"),
                    ("hawaiiens", "hawaïens"),
                    ("iraquien", "irakien"),
                    ("iraquiens", "irakiens"),
                    ("isle", "île"),
                    ("isles", "îles"),
                    ("khôl", "kohl"),
                    ("kohol", "kohl"),
                    ("koheul", "kohl"),
                    ("laïc", "laïque"),
                    ("laïcs", "laïques"),
                    ("lettonne", "lettone"),
                    ("lettonnes", "lettones"),
                    ("lis", "lys"),
                    ("nénuphar", "nénufar"),
                    ("nénuphars", "nénufars"),
                    ("ognon", "oignon"),
                    ("ognons", "oignons"),
                    ("orang-outan", "orang-outang"),
                    ("orangs-outans", "orangs-outangs"),
                    ("parafe", "paraphe"),
                    ("parafes", "paraphes"),
                    ("paye", "paie"),
                    ("payes", "paies"),
                    ("phantasme", "fantasme"),
                    ("phantasmes", "fantasmes"),
                    ("pizzéria", "pizzeria"),
                    ("pizzérias", "pizzerias"),
                    ("rapeur", "rappeur"),
                    ("rapeurs", "rappeurs"),
                    ("rencard", "rancard"),
                    ("rencards", "rancards"),
                    ("resurgir", "ressurgir"),
                    ("soûl", "saoul"),
                    ("soûls", "saouls"),
                    ("tannin", "tanin"),
                    ("tannins", "tanins"),
                    ("tartufe", "tartuffe"),
                    ("tartufes", "tartuffes"),
                    ("trimballer", "trimbaler"),
                    ("tzar", "tsar"),
                    ("tzars", "tsars"),
                    ("tzigane", "tsigane"),
                    ("tziganes", "tsiganes"),
                    ("ululer", "hululer"),
                    ("vantail", "ventail"),
                    ("yoghourt", "yogourt"), # yaourt
                    ("yoghourts", "yogourts"), # yaourt
                ]]

if __name__ == "__main__":

    import sys, os
    if len(sys.argv) == 2 and os.path.isfile(sys.argv[1]):
        with open(sys.argv[1], "r") as f:
            text = f.read()
            for line in text.splitlines():
                print(format_text_latin(line))
    else:
        print(format_text_latin(" ".join(sys.argv[1:])))