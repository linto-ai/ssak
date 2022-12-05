import os

from linastt.utils.text import format_text_fr

from .utils import Test

class TestFormatText(Test):

    def test_format_digits(self):
        self.assertEqual(
            format_text_fr("Ma grand-mère version 2.0 ne connaît pas wav2vec. Commande1"),
            "ma grand-mère version deux point zéro ne connaît pas wav to vec commande un"
        )
        self.assertEqual(
            format_text_fr("Elle pesait 67,5kg en 1995, et + en 1996."),
            "elle pesait soixante-sept virgule cinq kilogrammes en mille neuf cent quatre-vingt-quinze et plus en mille neuf cent quatre-vingt-seize"
        )
        self.assertEqual(
            format_text_fr("¾ et 3/4 et 3 m² et 3m³ et 3.7cm² et 3,7 cm³"),
            "trois quarts et trois quarts et trois mètres carrés et trois mètres cubes et trois point sept centimètres carrés et trois virgule sept centimètres cubes"
        )
        self.assertEqual(
            format_text_fr("10 000 et 100 000 000 ne sont pas pareils que 06 12 34 56 78"),
            "dix mille et cent millions ne sont pas pareils que zéro six douze trente-quatre cinquante-six soixante-dix-huit"
        )

        self.assertEqual(
            format_text_fr("La Rochelle - CAUE 17 2003 2005 Inventaire du Pays"),
            "la rochelle caue dix-sept deux mille trois deux mille cinq inventaire du pays"
            )

        self.assertEqual(
            format_text_fr("Elle emploie 147 collaborateurs et dispose de 22 000 m² de surfaces d' entreposage situées Casablanca-Oukacha ( 4 000 m² ) , Had Soualem ( 4 100 m² ) et Ain Sebaâ ( 14 000m² )"),
            "elle emploie cent quarante-sept collaborateurs et dispose de vingt-deux mille mètres carrés de surfaces d' entreposage situées casablanca-oukacha quatre mille mètres carrés had soualem quatre mille cent mètres carrés et ain sebaâ quatorze mille mètres carrés"
        )

        self.assertEqual(
            format_text_fr("1 200 et 01 2002"),
            "mille deux cents et zéro un deux mille deux"
        )

        self.assertEqual(
            format_text_fr("1 200 et 01 2002"),
            "mille deux cents et zéro un deux mille deux"
        )

        self.assertEqual(
            format_text_fr("1/6 000 de seconde pour les  quelques 6 000 précieuses"),
            "un six millième de seconde pour les quelques six mille précieuses"
        )

        self.assertEqual(
            format_text_fr("155/70-12 155/70-15 140/80-13"),
            #"cent cinquante-cinq soixante-dixièmes douze mille cent cinquante-cinq soixante-dix quinze mille cent quarante quatre-vingts treize"
            "cent cinquante-cinq soixante-dixièmes douze cent cinquante-cinq soixante-dixièmes quinze mille cent quarante quatre-vingts treize"
        )

        self.assertEqual(
            format_text_fr("Jeu 1 : 160/65-315  155/70-12 155/70-15"),
            "jeu un cent soixante soixante-cinq mille trois cent quinzièmes cent cinquante-cinq soixante-dixièmes douze cent cinquante-cinq soixante-dixièmes quinze"
        )

        self.assertEqual(
            format_text_fr("98-151/16, 39-1/16"),
            "quatre-vingt-dix-huit mille cent cinquante et un seize trente-neuf un seizième"
        )

        self.assertEqual(
            format_text_fr("Une quantité c' est un volume  Quand on divise un volume par une surface on obtient une hauteur Par exemple un 1 litre / m² = 1 000 000 mm3/1 000 000 mm² = 1mm"),
            "une quantité c' est un volume quand on divise un volume par une surface on obtient une hauteur par exemple un un litre mètres carrés un million millimètres trois un million millimètres carrés un millimètres"
        )

        self.assertEqual(
            format_text_fr("2222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222"),
            "infinité"
        )

    def test_format_special_chars(self):
        self.assertEqual(
            format_text_fr("L’état “from scratch”."),
            "l' état from scratch"
        )

    def test_websites(self):
        self.assertEqual(
            format_text_fr("http://www.linagora.blah.com/page.html est un site / www.linagora.com en est un autre"),
            "http deux points slash slash www point linagora point blah point com slash page point html est un site www point linagora point com en est un autre"
        )

        self.assertEqual(
            format_text_fr("www.len.com len.. ralenti"),
            "www point len point com len ralenti"
        )

    def test_abbr(self):
        self.assertEqual(
            format_text_fr("M. Dupont, Mme. Dupont, Mlle. Dupont"),
            "monsieur dupont madame dupont mademoiselle dupont"
        )
        self.assertEqual(
            format_text_fr("C'est la <DATE> est au format aaaa-mm-dd. ça mesure 3mm"),
            "c' est la date est au format aaaa-mm-dd ça mesure trois millimètres"
        )

    def test_spelling(self):
        self.assertEqual(
            format_text_fr("J'ai mis les clefs dans la serrure en revenant du bistrot, etc. Je suis un feignant et cætera."),
            "j' ai mis les clés dans la serrure en revenant du bistro et cetera je suis un fainéant et cetera"
        )

    def test_misc(self):
        self.assertEqual(
            format_text_fr("chef-d'oeuvre et- voilà - quoi"),
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
