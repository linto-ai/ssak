import re

_corrections_caracteres_speciaux_fr = [(re.compile('%s' % x[0]), '%s' % x[1])
                  for x in [
                    ("á","a"),
                    ("ä","a"),
                    ("å","a"),
                    ("Å","A"),
                    ("í","i"),
                    ("ö","o"),
                    ("ó","o"),
                    ("ò","o"),
                    ("ø","o"),
                    ("Ø","O"),
                    ("Ö","O"),
                    ("ǜ","u"),
                    ("ü","u"),
                    ("ú","u"),
                    ("ý","y"),
                    ("ÿ","y"),
                    ("ñ","n"),
                    ("ß","ss"),
                    ("ð","d"),
                    ("þ","z"), # utilisée pour transcrire le son d'une consonne fricative dentale sourde (comme le « th » de « thick » en anglais moderne)
                ]]