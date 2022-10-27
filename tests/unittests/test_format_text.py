from audiotrain.utils.text import format_text_fr

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

    def test_abbr(self):
        self.assertEqual(
            format_text_fr("M. Dupont, Mme. Dupont, Mlle. Dupont"),
            "monsieur dupont madame dupont mademoiselle dupont"
        )
        self.assertEqual(
            format_text_fr("C'est la <DATE> est au format aaaa-mm-dd. ça mesure 3mm"),
            "c' est la date est au format aaaa-mm-dd ça mesure trois millimètres"
        )