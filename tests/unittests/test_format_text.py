import os

from linastt.utils.text import format_text_latin, format_text_ar

from .utils import Test

class TestFormatTextLatin(Test):

    def test_format_digits(self):
        self.assertEqual(
            format_text_latin("Ma grand-mère version 2.0 ne connaît pas wav2vec. Commande1"),
            "ma grand-mère version deux point zéro ne connaît pas wav to vec commande un"
        )
        self.assertEqual(
            format_text_latin("Elle pesait 67,5kg en 1995, et + en 1996."),
            "elle pesait soixante-sept virgule cinq kilogrammes en mille neuf cent quatre-vingt-quinze et plus en mille neuf cent quatre-vingt-seize"
        )
        self.assertEqual(
            format_text_latin("¾ et 3/4 et 3 m² et 3m³ et 3.7cm² et 3,7 cm³"),
            "trois quarts et trois quarts et trois mètres carrés et trois mètres cubes et trois point sept centimètres carrés et trois virgule sept centimètres cubes"
        )
        self.assertEqual(
            format_text_latin("10 000 et 100 000 000 ne sont pas pareils que 06 12 34 56 78"),
            "dix mille et cent millions ne sont pas pareils que zéro six douze trente-quatre cinquante-six soixante-dix-huit"
        )

        self.assertEqual(
            format_text_latin("10,000 et 100,000,000 et 1111,000,000"),
            "dix mille et cent millions et un milliard cent onze millions"
        )

        self.assertEqual(
            format_text_latin("La Rochelle - CAUE 17 2003 2005 Inventaire du Pays"),
            "la rochelle caue dix-sept deux mille trois deux mille cinq inventaire du pays"
            )

        self.assertEqual(
            format_text_latin("Elle emploie 147 collaborateurs et dispose de 22 000 m² de surfaces d' entreposage situées Casablanca-Oukacha ( 4 000 m² ) , Had Soualem ( 4 100 m² ) et Ain Sebaâ ( 14 000m² )"),
            "elle emploie cent quarante-sept collaborateurs et dispose de vingt-deux mille mètres carrés de surfaces d' entreposage situées casablanca-oukacha quatre mille mètres carrés had soualem quatre mille cent mètres carrés et ain sebaâ quatorze mille mètres carrés"
        )

        self.assertEqual(
            format_text_latin("1 200 et 01 2002"),
            "mille deux cents et zéro un deux mille deux"
        )

        self.assertEqual(
            format_text_latin("1 200 et 01 2002"),
            "mille deux cents et zéro un deux mille deux"
        )

        self.assertEqual(
            format_text_latin("1/6 000 de seconde pour les  quelques 6 000 précieuses"),
            "un six millième de seconde pour les quelques six mille précieuses"
        )

        self.assertEqual(
            format_text_latin("155/70-12 155/70-15 140/80-13"),
            #"cent cinquante-cinq soixante-dixièmes douze mille cent cinquante-cinq soixante-dix quinze mille cent quarante quatre-vingts treize"
            "cent cinquante-cinq soixante-dixièmes douze cent cinquante-cinq soixante-dixièmes quinze mille cent quarante quatre-vingts treize"
        )

        self.assertEqual(
            format_text_latin("Jeu 1 : 160/65-315  155/70-12 155/70-15"),
            "jeu un cent soixante soixante-cinq mille trois cent quinzièmes cent cinquante-cinq soixante-dixièmes douze cent cinquante-cinq soixante-dixièmes quinze"
        )

        self.assertEqual(
            format_text_latin("98-151/16, 39-1/16"),
            "quatre-vingt-dix-huit mille cent cinquante et un seize trente-neuf un seizième"
        )

        self.assertEqual(
            format_text_latin("Attentats du 21//07: le suspect extradé d'Italie inculpé à Londres. Nous sommes le 01/01/2019 à 10h00 Il fait -1°C."),
            "attentats du vingt et un juillet le suspect extradé d' italie inculpé à londres nous sommes le premier janvier deux mille dix-neuf à dix heures il fait moins un degrés celsius"
        )

        self.assertEqual(
            format_text_latin("32/01/2019 31/13 "),
            "trente-deux zéro un deux mille dix-neuf trente et un treizièmes"
        )

        self.assertEqual(
            format_text_latin("Une quantité c' est un volume  Quand on divise un volume par une surface on obtient une hauteur Par exemple un 1 litre / m² = 1 000 000 mm3/1 000 000 mm² = 1mm"),
            "une quantité c' est un volume quand on divise un volume par une surface on obtient une hauteur par exemple un un litre mètres carrés un million millimètres trois un million millimètres carrés un millimètres"
        )

        text = "2222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222"
        expected = text.replace("2", "deux ").strip()
        self.assertEqual(
            format_text_latin(text),
            expected
        )

        text = "-1900190119021903190419051906190719081909191019111912191319141915191619171918191919201921192219231924192519261927192819291930193119321933193419351936193719381939194019411942194319441945194619471948194919501951195219531954195519561957195819591960196119621963196419651966196719681969197019711972197319741975197619771978197919801981198219831984198519861987198819891990199119921993199419951996199719981999200020012002200320042005200620072008200920102011201220132014201520162017"
        self.assertEqual(
            format_text_latin(text),
            "moins "+" ".join([format_text_latin(x) for x in text[1:]])
        )

    def test_format_currencies(self):

        for text in [
            "ça coute 1,20€",
            "ça coute 1,20 €",
        ]:
            self.assertEqual(
                format_text_latin(text),
                "ça coute un euros vingt" # TODO: remove the s...
            )

        for text in [
            "ça coute 1,20$",
            "ça coute 1,20 $",
        ]:
            self.assertEqual(
                format_text_latin(text),
                "ça coute un dollars vingt" # TODO: remove the s...
            )

        for text in [
            "ça coute 1.20$ * 6 mois",
            "ça coute 1.20 $ * 6 mois",
        ]:
            self.assertEqual(
                format_text_latin(text),
                "ça coute un point vingt dollars six mois"
            )

    def test_format_special_chars(self):
        self.assertEqual(
            format_text_latin("L’état “from scratch”."),
            "l' état from scratch"
        )

        self.assertEqual(
            format_text_latin("``Elle veut nous faire croire qu'elle était complètement abrutie'', a affirmé l'avocat général notant cependant que l'ancienne prostituée -qui aurait dit dans la voiture ``Vas-y,allume-les!''-avait réussi sa reconversion après 1994."),
            "elle veut nous faire croire qu' elle était complètement abrutie a affirmé l' avocat général notant cependant que l' ancienne prostituée qui aurait dit dans la voiture vas-y allume-les avait réussi sa reconversion après mille neuf cent quatre-vingt-quatorze"
        )

        self.assertEqual(
            format_text_latin("Hello aaaf-bearn-gascogne, belle-mère, rouge-gorge."),
            "hello aaaf bearn gascogne belle-mère rouge-gorge"
        )

    def test_websites(self):
        self.assertEqual(
            format_text_latin("http://www.linagora.blah.com/page.html est un site / www.linagora.com en est un autre"),
            "http deux points slash slash www point linagora point blah point com slash page point html est un site www point linagora point com en est un autre"
        )

        self.assertEqual(
            format_text_latin("www.len.com len.. ralenti"),
            "www point len point com len ralenti"
        )

    def test_abbr(self):
        self.assertEqual(
            format_text_latin("M. Dupont, Mme. Dupont, Mlle. Dupont"),
            "monsieur dupont madame dupont mademoiselle dupont"
        )
        self.assertEqual(
            format_text_latin("C'est la <DATE> est au format aaaa-mm-dd. ça mesure 3mm"),
            "c' est la date est au format aaaa millimètres dd ça mesure trois millimètres" # Not the best...
        )

    def test_spelling(self):
        self.assertEqual(
            format_text_latin("J'ai mis les clefs dans la serrure en revenant du bistrot, etc. Je suis un feignant et cætera."),
            "j' ai mis les clés dans la serrure en revenant du bistro et cetera je suis un fainéant et cetera"
        )

    def test_misc(self):
        self.assertEqual(
            format_text_latin("chef-d'oeuvre et- voilà - quoi"),
            "chef-d' oeuvre et voilà quoi"
        )


    def test_non_regression_fr(self):
        
        output_file = self.get_temp_path("output.txt")
        acronym_file = self.get_temp_path("acronyms.txt")
        special_char_file = self.get_temp_path("special_chars.txt")
        for f in [output_file, acronym_file, special_char_file]:
            if os.path.exists(f): os.remove(f)
        self.assertRun([
            self.get_tool_path("clean_text_fr.py"),
            self.get_data_path("text/frwac.txt"),
            output_file,
            "--extract_parenthesis",
            "--file_acro", acronym_file,
            "--file_special", special_char_file,
        ])
        self.assertNonRegression(output_file, "format_text/output.txt")
        self.assertNonRegression(acronym_file, "format_text/acronyms.txt")
        self.assertNonRegression(special_char_file, "format_text/special_chars.txt")
        for f in [output_file, acronym_file, special_char_file]:
            os.remove(f)

class TestFormatTextArabic(Test):

    def test_options(self):

        sentence = "في اللغة الإنجليزية ، يمكن للمرء أن يقول \"Hello world\"!"

        self.assertEqual(
            format_text_ar(sentence, keep_punc=False, keep_latin_chars=False),
            'في اللغة الإنجليزية يمكن للمرء أن يقول '
        )

        self.assertEqual(
            format_text_ar(sentence, keep_punc=True, keep_latin_chars=False),
            'في اللغة الإنجليزية ، يمكن للمرء أن يقول !'
        )

        self.assertEqual(
            format_text_ar(sentence, keep_punc=False, keep_latin_chars=True),
            'في اللغة الإنجليزية يمكن للمرء أن يقول Hello world '
        )

        self.assertEqual(
            format_text_ar(sentence, keep_punc=True, keep_latin_chars=True),
            'في اللغة الإنجليزية ، يمكن للمرء أن يقول Hello world !'
        )

    def test_format_digits(self):
        
        self.assertEqual(
            format_text_ar("بعض الأرقام: 01 و 314 و 315.5 و ۰۹ و ۹۰"),
            'بعض الأرقام صفر واحد و ثلاثمائة و أربعة عشر و ثلاثمائة و خمسة عشر فاصيله خمسة و صفر تسعة و تسعون '
        )

        self.assertEqual(
            format_text_ar("يوجد 10000 شخص ، عنوان IP الخاص بي هو 951.357.123 ، ورقم هاتفي هو 06 12 34 56 78"),
            'يوجد عشرة آلاف شخص عنوان الخاص بي هو تسعمائة و واحد و خمسون فاصيله ثلاثمائة و سبعة و خمسون مائة و ثلاثة و عشرون ورقم هاتفي هو صفر ستة اثنا عشر أربعة و ثلاثون ستة و خمسون ثمانية و سبعون '
        )
                
        self.assertEqual(
            format_text_ar("وأغلق بسعر $45.06 للبرميل."),
            'وأغلق بسعر دولار خمسة و أربعون فاصيله صفر ستة للبرميل '
        )

        self.assertEqual(
            format_text_ar("اليوم 22/03/2002"),
            'اليوم اثنتان و عشرون مارس ألفان و اثنان '
        )

    def test_symbols_converting(self):
        
        self.assertEqual(
            format_text_ar("للعام 1435/ 1436هـ"),
            'للعام واحد ألف و أربعمائة و خمس و ثلاثون واحد ألف و أربعمائة و ست و ثلاثون هجري '
        )

        self.assertEqual(
            format_text_ar("7 ق.م"),
            ' سبع قبل الميلاد '
        )

        self.assertEqual(
            format_text_ar("300¥ = 300$ = 300£ = 300₹"),
            ' ثلاثمائة ين يساوي ثلاثمائة دولار يساوي ثلاثمائة جنيه يساوي ثلاثمائة روبية هندية '
        )
