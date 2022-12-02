import re
import math
from num2words import num2words

from linastt.utils.text_utils import collapse_whitespace, remove_special_characters, text_unescape
from linastt.utils.misc import flatten

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

    text = collapse_whitespace(text)

    text = text.lower() # TCOF
    return text

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

def format_text_fr(text,
    lower_case = True,
    keep_punc = False,
    remove_ligatures = True,
    extract_parenthesis = False,
    fid_acronyms = None,
    fid_special_chars = None,
    safety_checks = True,
    ):

    opts = _rm_key(locals(), "text")

    text_orig = text

    try:

        # Recursive call (list)
        if isinstance(text, list):
            return [format_text_fr(t, **opts) for t in text]

        # Recursive call (line breaks)
        if "\n" in text:
            return "\n".join([format_text_fr(t, **opts) for t in text.split("\n")])
        
        # Recursive call (parenthesis)
        if extract_parenthesis and "(" in text and ")" in text:
            in_parenthesis = re.findall(r"\(([^\(\)]*?)\)", text)
            if len(in_parenthesis):
                in_parenthesis = [s.rstrip(")").lstrip("(") for s in in_parenthesis]
                regex = "("+")|(".join(["\("+text_unescape(p)+"\)" for p in in_parenthesis])+")"
                without_parenthesis = re.sub(regex, "", text)
                # assert without_parenthesis != text
                if without_parenthesis != text: # Avoid infinite recursion
                    texts = [without_parenthesis] + in_parenthesis
                    return "\n".join([format_text_fr(t, **opts) for t in texts])


        global _ALL_ACRONYMS

        if re.search(r"[IVX]", text):
            for k,v in _romans.items():
                text = re.sub(r"\b" + k + r"\b", v, text)

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

        for reg, replacement in _corrections_caracteres_speciaux_fr:
            text = re.sub(reg, replacement, text)


        text = ' '+text+' '

        numbers=re.findall("\d+[,.]000",text)
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
        text = re.sub("^ *-+", "", text)
        text = re.sub("\^+","", text)
        text = re.sub(" +(- +)+", " ", text)
        text = re.sub("- ", " ", text)
        text = re.sub("([a-zàâäçèéêëîïôùûü]+)- +", r"\1-", text)
        text = re.sub(" -([a-zàâäçèéêëîïôùûü]+)", r"-\1", text)
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

        chiffres = re.findall(r"\b1(?:ère|ere|er|re|r)|2(?:nd|nde)|\d+(?:ème|eme|e)\b", text)
        chiffres = sorted(list(set(chiffres)), reverse=True, key=lambda x: (len(x), x))
        for chiffre in chiffres:
            word = undigit(re.findall(r"\d+", chiffre)[0], to= "ordinal")
            text = re.sub(r'\b'+str(chiffre)+r'\b', word, text)

        text = re.sub(r"\b(\d+),(\d+)",r"\1 virgule \2", text)
        text = re.sub(r"\b(\d+)\.(\d+)\b",r"\1 point \2", text)
        text = re.sub(r'([a-z])2([a-z])', r'\1 to \2', text) # wav2vec -> wav to vec
        text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text) # space after digits
        text = re.sub(r'(\d)-', r'\1 ', text) # For things like 40-MFCC

        # Digits
        chiffres = re.findall(r"(?:\b[\d/]*\d+(?: \d\d\d)+\b)|(?:\d[/\d]*)",text)
        chiffres = list(map(lambda s: s.strip(r"[/ ]"), chiffres))
        chiffres = list(set(chiffres))
        chiffres = chiffres + flatten([c.split() for c in chiffres if " " in c])
        #chiffres = sorted(chiffres, reverse=True, key=lambda x: ("/" in x, len(x)))
        chiffres = sorted(chiffres, reverse=True, key=lambda x: (len(x), x))
        for chiffre in chiffres:
            numslash = len(re.findall("/", chiffre))
            if numslash == 0:
                word = undigit(chiffre)
            elif numslash == 1:
                i = chiffre.index("/")
                first = undigit(chiffre[:i])
                second = undigit(chiffre[i+1:], to="denominator")
                if float(chiffre[:i]) > 2. and second[-1] != "s":
                    second += "s"
                word = first + " " + second
            else:
                word = " / ".join([undigit(s) for s in chiffre.split('/') if s])
            if " " in chiffre:
                text = re.sub(r'\b'+str(chiffre)+r'\b', " "+word+" ", text)
            else:
                text = re.sub(str(chiffre), " "+word+" ", text)

        if safety_checks:
            if re.findall(r"\d", text):
                raise ValueError(f"Failed to convert all digits to words\nInput: {text_orig}\nOutput: {text}")

        # Fractions
        text = re.sub(r"½", " un demi ", text)
        text = re.sub(r"⅓", " un tiers ", text)
        text = re.sub(r"⅔", " deux tiers ", text)
        text = re.sub(r"¼", " un quart ", text)
        text = re.sub(r"¾", " trois quarts ", text)
        # Exponents
        text = re.sub(r"\bm²", " mètres carrés ", text)
        text = re.sub(r"\bm³", " mètres cubes ", text)
        text = re.sub(r"²", " carrés ", text)
        text = re.sub(r"³", " cubes ", text)
        text = re.sub(r"⁵", " puissance cinq ", text)
        text = re.sub(r"⁷", " puissance sept ", text)

        text = re.sub(" '", " ", text)
        text = re.sub('--+',' ', text)
        text = re.sub('_',' ', text)
        text = re.sub('–',' ', text)
        text = re.sub('—+',' ', text)
        text = re.sub('…','...', text)
        text = re.sub('\*+', ' ', text)
        text = re.sub(r"[«“][^\S\r\n]*", '"', text)
        text = re.sub(r"[^\S\r\n]*[»”″„]", '"', text)
        text = re.sub(r"[’‘‛]", "'", text)
        text = re.sub("‚", ",", text)
        text = re.sub(r"–", "-", text)
        text = re.sub('#+',' ', text)
        text = re.sub(" "," ",text)
        text = re.sub(' ', '  ',text)

        text = re.sub('\{|\}|\(|\)|\[|\]|"|=',' ',text)
        text = re.sub('(\.|\?|\!|,|;|:)-',r'\1 ', text)

        for reg, replacement in _corrections_abbreviations_fr:
            text = re.sub(reg, replacement, text)

        for reg, replacement in _multi_spelling_words:
            text = re.sub(reg, replacement, text)

        # Symbols
        text = re.sub(r"°c\b", "degrés", text)
        text = re.sub("°", "degrés", text)
        text = re.sub("&"," et ", text)
        text = re.sub('%', ' pour cent ', text)
        text = re.sub('‰', ' pour mille ', text)
        text = re.sub("~"," environ ", text)
        text = re.sub("µ"," micro ", text)
        text = re.sub("μ"," micro ", text)
        text = re.sub("§"," paragraphe ", text)
        text = re.sub(r"[\+⁺]"," plus ", text)
        text = re.sub(r"⁻"," moins ", text)
        text = re.sub("±"," plus ou moins ", text)
        text = re.sub(r"ᵉʳ","er", text)
        text = re.sub(r"ᵉ","e", text)
        text = re.sub("·","", text)
        # Currencies (TODO: decide plural or singular, manage 1.30 €)
        text = re.sub('€', ' euros ', text)
        text = re.sub('¥', ' yens ', text)
        text = re.sub('£', ' livres ', text)
        text = re.sub('\$', ' dollars ', text)
        text = re.sub("¢"," cents ", text)

        if not keep_punc:
            text = re.sub(r',|;|:|\!|\?|/|\.',' ',text)

        text = re.sub(' - | -$|^- ','', text)

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

def undigit(str, to="cardinal", lang = "fr"):
    str = re.sub(" ","", str)
    if to == "denominator":
        assert lang == "fr"
        if str == "2": return "demi"
        if str == "3": return "tiers"
        if str == "4": return "quart"
        to = "ordinal"
    if str.startswith("0") and to == "cardinal":
        numZeros = len(re.findall(r"0+", str)[0])
        if numZeros < len(str):
            return numZeros * (my_num2words(0, lang=lang, to="cardinal")+" ") + my_num2words(float(str), lang=lang, to=to)
    return my_num2words(float(str), lang=lang, to=to)

def my_num2words(x, lang = "fr", to = "cardinal"):
    """
    Bugfix for num2words
    """
    try:
        if lang == "fr" and to == "ordinal":
            return num2words(x, lang=lang, to=to).replace("vingtsième", "vingtième")
        else:
            return num2words(x, lang=lang, to=to)
    except OverflowError:
        if x == math.inf: # !
            return "infinité"
        # TODO: print a warning
        return my_num2words(x//10, lang=lang, to=to)


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
                    ("ħ","h"),                    
                    ("ĵ","j"),
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

_romans = {
    'I': 'un',
    'II': 'deux',
    'III': 'trois',
    'IV': 'quatre',
    'VII': 'sept',
    'VIII': 'huit',
    'IX': 'neuf',
    'XI': 'onze',
    'XII': 'douze',
    'XIII': 'treize',
    'XIV': 'quatorze',
    'XV': 'quinze',
    'XVI': 'seize',
    'XVII': 'dix-sept',
    'XVIII': 'dix-huit',
    'XIX': 'dix-neuf',
    'XX': 'vingt',
    'XXI': 'vingt-et-un',
    'XXII': 'vingt-deux',
    'Ier': 'premier',
    'Iere': 'première',
    'Ière': 'première',
    'IIe': 'deuxième',
    'IIIe': 'troisième',
    'IVe': 'quatrième',
    'VIIe': 'septième',
    'VIIIe': 'huitième',
    'IXe': 'neuvième',
    'XIe': 'onzième',
    'XIIe': 'douzième',
    'XIIIe': 'treizième',
    'XIVe': 'quatorzième',
    'XVe': 'quinzième',
    'XVIe': 'seizième',
    'XVIIe': 'dix-septième',
    'XVIIIe': 'dix-huitième',
    'XIXe': 'dix-neuvième',
    'XXe': 'vingtième',
    'XXIe': 'vingt-et-unième',
    'XXIIe': 'vingt-deuxième',
}

if __name__ == "__main__":

    import sys, os
    if len(sys.argv) == 2 and os.path.isfile(sys.argv[1]):
        with open(sys.argv[1], "r") as f:
            text = f.read()
            for line in text.splitlines():
                print(format_text_fr(line))
    else:
        print(format_text_fr(" ".join(sys.argv[1:])))