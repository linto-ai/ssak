#!/usr/bin/env python3

"""scrap.py: Scrape information websites to extract text."""

__author__      = "Jerome Louradour"
__copyright__   = "Copyright 2022, Linagora"


import os

import itertools
import re
import time

from slugify import slugify

from bs4 import BeautifulSoup # python -m pip install beautifulsoup4
from selenium import webdriver # python -m pip install selenium
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common import exceptions as selenium_exceptions

VERBOSE_DEFAULT = 1

def scrape_20minutes(outfolder, verbose = VERBOSE_DEFAULT, max_pages = None, close_at_the_end = True):

    return scrape(
        "https://www.20minutes.fr/",
        [
            {"id": "didomi-notice-agree-button"}, # Accept cookies
            {"id": "open-menu-button"}, # Click on the Menu button to see all categories
        ],
        ("ul", "menu-list"),
        [("a",), ("article", {"class_":"box preview list-item preview-stretch"})],
        [None],
        outfolder, verbose = verbose, max_pages = max_pages,
        close_at_the_end = close_at_the_end,
    )

def scrape_huffingtonpost(outfolder, verbose = VERBOSE_DEFAULT, max_pages = None, close_at_the_end = True):

    return scrape(
        "https://www.huffingtonpost.fr/",
        [
            {"class": "gdpr-hfp-button gdpr-hfp-button--big gdpr-hfp-button--main"}, # Accept cookies
            {"id": "batchsdk-ui-alert__buttons_negative"}, # Refuse to receive information
        ],
        ("a", "subNavLeft-categoriesTitle"), # Top menu
        [("a", {"class_": "item-image"})], # Articles in a page
        [['article-chapo'], ['asset', 'asset-text']], # Paragraph classes in articles
        outfolder, verbose = verbose, max_pages = max_pages,
        #button_sleep = 1,
        close_at_the_end = close_at_the_end,
    )

def scrape_leparisien(outfolder, verbose = VERBOSE_DEFAULT, max_pages = None, close_at_the_end = True):

    return scrape(
        "https://www.leparisien.fr/",
        [
            {"id": "didomi-notice-agree-button"}, # Accept cookies
        ],
        ("a", "link_lg no-decorate padding_right_sm no-letter-spacing color_black"), # Top menu
        [("a", {"class_": "no-decorate no-active title_sm"})], # Articles in a page
        [['paragraph', 'text_align_left']], # Paragraph classes in articles
        outfolder, verbose = verbose, max_pages = max_pages,
        ignore_article_if = lambda x: x.text.startswith("Abonnés"),
        close_at_the_end = close_at_the_end,
    )

def scrape_actu(outfolder, verbose = VERBOSE_DEFAULT, max_pages = None, close_at_the_end = True):

    return scrape(
        "https://actu.fr/",
        [
            {"id": "didomi-notice-agree-button"}, # Accept cookies
        ],
        ("a", ""), # Top menu
        [("a")], # Articles in a page
        [None], # Paragraph classes in articles
        outfolder, verbose = verbose, max_pages = max_pages,
        ignore_article_if = lambda x: x.get("data-trk") is None,
        close_at_the_end = close_at_the_end,
    )

def scrape_numerama(outfolder, verbose = VERBOSE_DEFAULT, max_pages = None, close_at_the_end = True):
    
    return scrape(
        "https://www.numerama.com/",
        [
            {"class": "sd-cmp-JnaLO"}, # Accept cookies
        ],
        ("a", "is-hidden-menu-open is-flex is-align-items-center is-active"), # Top Menu
        None, # Articles in a page
        [None], # Paragraph classes in articles
        outfolder, verbose = verbose, max_pages = max_pages,
        close_at_the_end = close_at_the_end,
    )

def scrape_lemonde(outfolder, verbose = VERBOSE_DEFAULT, max_pages = None, close_at_the_end = True):
    
    return scrape(
        "https://www.lemonde.fr/",
        [
            {"class": "gdpr-lmd-button gdpr-lmd-button--big gdpr-lmd-button--main"}, # Accept cookies
        ],
        ("a", "js-actu-tag"), # Top Menu
        [("a", {"class": "teaser__link"})],  # Articles in a page
        [['post__live-container--answer-text', 'post__space-node'], ['article__paragraph'], ['article__paragraph', 'article__paragraph--lf']], # Paragraph classes in articles
        outfolder, verbose = verbose, max_pages = max_pages,
        ignore_article_if = lambda x: x.text.lstrip().startswith("Article réservé à nos abonnés"),
        close_at_the_end = close_at_the_end,
    )

def scrape_nouvelobs(outfolder, verbose = VERBOSE_DEFAULT, max_pages = None, close_at_the_end = True):

    return scrape(
        "https://www.nouvelobs.com/",
        [
            {"class": "gdpr-glm-button gdpr-glm-button--standard"}, # Accept cookies
        ],
        ("a", "menu__main"), # Top Menu
        None,  # Articles in a page
        [None], # Paragraph classes in articles
        outfolder, verbose = verbose, max_pages = max_pages,
        close_at_the_end = close_at_the_end,
    )

def scrape_lesechos(outfolder, verbose = VERBOSE_DEFAULT, max_pages = None, close_at_the_end = True):

    def is_top_menu(x):
        if x.get("/") == "/":
            return False
        classe = x.get("class")
        return isinstance(classe, list) and len(classe) > 3 and classe[:3] == ['sc-1560xb1-0', 'loUvLB', 'sc-ztp7xd-0']

    def is_article(x):
        classe = x.get("class")
        return isinstance(classe, list) and len(classe) > 3 and classe[:3] == ['sc-1560xb1-0', 'loUvLB', 'sc-iphbbg-1']

    return scrape(
        "https://www.lesechos.fr/",
        [
            {"id": "didomi-notice-agree-button"}, # Accept cookies
        ],
        ("a", is_top_menu), # Top Menu
        [("a", is_article)],  # Articles in a page
        [['sc-14kwckt-6', 'gPHWRV']], # Paragraph classes in articles
        outfolder, verbose = verbose, max_pages = max_pages,
        close_at_the_end = close_at_the_end,
    )

def scrape_slate(outfolder, verbose = VERBOSE_DEFAULT, max_pages = None, close_at_the_end = True):

    categories = [
        #'audio',
        'boire-manger',
        'culture',
        'economie',
        'egalites',
        'enfants',
        'grand-format',
        'medias',
        'monde',
        'politique',
        'sante',
        'sciences',
        'societe',
        'sports',
        'story',
        'studio',
        'tech-internet']

    def is_article(x):
        link = x.get("href", "")
        for cat in [
            #'audio',
            'boire-manger',
            'culture',
            'economie',
            'egalites',
            'enfants',
            'grand-format',
            'medias',
            'monde',
            'politique',
            'sante',
            'sciences',
            'societe',
            'sports',
            'story',
            'studio',
            'tech-internet']:
            if link.startswith("/" + cat + "/") and len(link) > len(cat) + 2:
                return True
        return False

    return scrape(
        "https://www.slate.fr/",
        [
            {"class": "sd-cmp-JnaLO"}, # Accept cookies
        ],
        categories, # Top Menu
        [("a", is_article)],  # Articles in a page
        [None, ['Corps']], # Paragraph classes in articles
        outfolder, verbose = verbose, max_pages = max_pages,
        close_at_the_end = close_at_the_end,
    )

# def scrape_lefigaro(outfolder, verbose = VERBOSE_DEFAULT, max_pages = None):
    
#     return scrape(
#         "https://www.lefigaro.fr/",
#         [
#             ???, # Accept cookies
#         ],
#         ..., # Top Menu
#         [...],  # Articles in a page
#         [...], # Paragraph classes in articles
#         outfolder, verbose = verbose, max_pages = max_pages,
#         button_sleep = 1,
#     )


def norm_text(text):
    return re.sub("\n+"," ", text).strip()


def click_button(driver, *kargs, verbose = True, **kwargs):
    s = ",".join(["@"+k for k in kargs])+",".join([f'@{k.rstrip("_")}="{v}"' for k,v in kwargs.items()])
    if verbose>1:
        print("* Click on:", s)
    WebDriverWait(driver, 0).until(EC.element_to_be_clickable((By.XPATH,f'//*[{s}]'))).click()

def absolute_link(sublink, base_url):
    if sublink.startswith('//www.'):
        sublink = "https:" + sublink
    if sublink.startswith('./'):
        sublink = sublink[1:]
    if sublink.startswith('/'):
        sublink = sublink.lstrip("/")
        sublink = base_url.rstrip("/") + "/" + sublink
    return sublink

def scrape(
    website,
    click_buttons,
    category_section,
    article_section,
    paragraph_classes,
    outfolder,
    button_sleep = 0,
    sleep = 0,
    max_pages = None,
    verbose = True,
    ignore_article_if = None,
    close_at_the_end = True,
    ):


    driver = webdriver.Firefox()

    text_to_discard = [ norm_text(t) for t in [
        "\nChoix de consentement © Copyright 20\xa0Minutes  - La fréquentation de 20\xa0Minutes est certifiée par l’ACPM \n",
        "À voir également sur Le HuffPost :",
        "Suivez l'actualité de vos communes favorites dans l'onglet Mon actu",
        "Commentaires",
        "Step 1", "Step 2",
        "À LIRE AUSSI",
    ]]
    # For websites like actu.fr with little structure
    text_start_to_discard = [
        "Cet article vous a été utile",
        "À voir ",
        "À lire ",
        "Source ",
        "Aïe vous avez refusé les cookies",
        "Partagez",
        "Newsletter Actu",
        "1\xa0 ",
        "2\xa0 ",
        "3\xa0 ",
        "4\xa0 ",
        "5\xa0 ",
        "On aime ",
        "Crédit photo ",
        "Certains liens de cet article ",
        "Vous pouvez retrouver ",
        "---",
        "Cliquer ici ",
        "=> Cliquer ici ",
        "AFP/",
        "REUTERS/",
    ]
    collected_text = []

    num_pages = 0

    try:
        # Go to main page
        driver.get(website)

        for k in click_buttons:
            try:
                if isinstance(k, dict):
                    click_button(driver, **k, verbose = verbose)
                elif isinstance(k, (tuple, list)):
                    click_button(driver, *k, verbose = verbose)
                else:
                    click_button(driver, k, verbose = verbose)
                if button_sleep:
                    if verbose>1:
                        print("* Sleeping", button_sleep, "seconds")
                    time.sleep(button_sleep)
            except selenium_exceptions.TimeoutException:
                print("WARNING: Error when clicking on button", k)

        os.makedirs(outfolder, exist_ok=True)

        # Get all the sections
        soup = BeautifulSoup(driver.page_source, "html.parser")

        assert article_section is None or len(article_section) <= 2
        double_embedding = article_section is not None and len(article_section)>1 
        
        
        found_sections = False
        found_subsections = False
        found_articles = False
        if category_section is None:
            sections = [{
                "class": "",
                "href": website
            }]
        elif isinstance(category_section, list) and len(category_section) > 2:
            sections = [{"class": "", "href": absolute_link("/"+c+"/", website)} for c in category_section]
        elif callable(category_section[1]):
            sections = [x for x in soup.find_all(category_section[0]) if category_section[1](x)]
        elif isinstance(category_section[1], str):
            sections = soup.find_all(category_section[0], class_=category_section[1])
        else:
            raise RuntimeError("Unknown category_section", category_section)
        for section in sections:
            if section.get("class") is None: # To differentiate class="" from class missing
                continue
            found_sections = True
            found_subsections = False
            if verbose>1 and double_embedding:
                links = section.find_all(article_section[0][0], **(article_section[0][1] if len(article_section[0])>1 else {}))
            else:
                links = [section]
            for link in links:
                found_subsections = True
                found_articles = False
                if verbose>1:
                    print("* Reading link:", link)
                if not link.get('href'):
                    print("WARNING: no href in link", link)
                    continue
                link = link.get('href')
                link = absolute_link(link, website)

                # Go to the section
                driver.get(link)

                # Get all the articles
                soup = BeautifulSoup(driver.page_source, "html.parser")

                
                if article_section is None:
                    # Default simple strategy
                    articles = soup.find_all("a", {"href": lambda x: x and x.endswith("html")})
                else:
                    larticle_section = article_section[-1]
                    if verbose>1:
                        print("* Reading section:", larticle_section)
                    if len(larticle_section) == 2 and callable(larticle_section[1]):
                        articles = [x for x in soup.find_all(larticle_section[0]) if larticle_section[1](x)]
                    else:
                        articles = soup.find_all(larticle_section[0], **larticle_section[1] if len(larticle_section)>1 else {})

                for article in articles:

                    if ignore_article_if is not None and ignore_article_if(article):
                        continue

                    found_articles = True
                    sublink = article.get("href")
                    if not sublink:
                        sublink = article.find_all("a")[0]["href"]

                    sublink = absolute_link(sublink, (link if double_embedding else website))
                    start = website.rstrip("/")+"/"
                    assert sublink.startswith(start)
                    title = sublink[len(start):]

                    filename = os.path.join(outfolder, title+".txt")
                    if os.path.isfile(filename):
                        continue
                    if sleep: time.sleep(sleep)
                    removeit = False
                    file = None

                    # Go to the article
                    driver.get(sublink)

                    # Get the text
                    soup = BeautifulSoup(driver.page_source, "html.parser")
                    found_paragraphs = False
                    for paragraph in soup.find_all('p'):
                        if paragraph.get('class') not in paragraph_classes:
                            continue
                        text = norm_text(paragraph.text)
                        if "Une page 404 est une page renvoyée par le serveur d’un site" in text:
                            removeit = True
                            break
                        if not text or text in text_to_discard:
                            continue
                        if text.endswith(" …"): # Unfinished sentence
                            continue
                        skip = False
                        for s in text_start_to_discard:
                            if text.startswith(s):
                                skip = True
                                break
                        if skip:
                            continue
                        if text in collected_text:
                            print("WARNING: text already collected:", text)
                            continue
                        collected_text.append(text)
                        if file is None:
                            if not os.path.isdir(os.path.dirname(filename)):
                                os.makedirs(os.path.dirname(filename))
                            file = open(filename, "w")
                        print(text, file = file)
                        if verbose and not found_paragraphs:
                            print(filename)
                        found_paragraphs = True
                    if file is not None:
                        file.close()
                        if removeit:
                            print("PROBLEM: Removing file")
                            os.remove(filename)
                        else:
                            num_pages += 1
                            if max_pages and num_pages >= max_pages:
                                if close_at_the_end:
                                    driver.close()
                                return
                    if not found_paragraphs:
                        print("WARNING: no paragraph found in", sublink)
                if not found_articles:
                    print("WARNING: No articles found ({article_section})")
            if not found_subsections:
                raise RuntimeError(f"No subsections found with {article_section}")
        if not found_sections:
            raise RuntimeError(f"No sections found with {category_section}")
        #driver.close()

    except Exception as e:
        try:
            if not found_subsections:
                soup = section
            else:
                soup = BeautifulSoup(driver.page_source, "html.parser")
            filename = os.path.realpath("error.html")
            print("ERROR: Writing current page to", filename)
            print(soup, file = open(filename, "w"))
        except: pass
        try:
            print("ERROR with URL: ", driver.current_url)
            if driver.current_url == website:
                # Print clickable buttons
                for button in driver.find_elements(by = By.TAG_NAME, value = "button"):
                    print("Button -- class =", button.get("class"), "id = ", button.get("id"))
        except: pass
        print(e)
        #raise e

    finally:
        if close_at_the_end:
            driver.close()


if __name__ == "__main__":

    print("Note that you can disable graphical interface by doing:\n\
```\n\
#install Xvfb\n\
sudo apt-get install xvfb\n\
\n\
#set display number to :99\n\
Xvfb :99 -ac &\n\
export DISPLAY=:99\n\
```")

    import argparse

    parser = argparse.ArgumentParser(description='Scrape a website.')
    parser.add_argument('output', type=str, default="data", help='Output folder', nargs="?")
    parser.add_argument("--no-close", action="store_true", help="Do not close the browser at the end", default=False)
    parser.add_argument("--loop", action="store_true", help="Do loop infinity", default=False)
    parser.add_argument("--verbose", action="store_true", help="More verbose", default=False)
    args = parser.parse_args()

    outputfolder = args.output

    kwargs = {"close_at_the_end" : not args.no_close, "verbose" : 2 if args.verbose else 1}

    # Scrape in a random order
    import random
    indices = list(range(9))
    random.shuffle(indices)

    if args.loop:
        def iters():
            while True:
                yield
    else:
        def iters():
            return [None]

    for iter in iters():

        for i in indices:
            if i==0:
                scrape_huffingtonpost(outputfolder+"/huffingtonpost", **kwargs)
            elif i==1:
                scrape_20minutes(outputfolder+"/20minutes", **kwargs)
            elif i==2:
                scrape_leparisien(outputfolder+"/leparisien", **kwargs)
            elif i==3:
                scrape_actu(outputfolder+"/actu", **kwargs)
            elif i==4:
                scrape_numerama(outputfolder+"/numerama", **kwargs)
            elif i==5:
                scrape_lemonde(outputfolder+"/lemonde", **kwargs)
            elif i==6:
                scrape_nouvelobs(outputfolder+"/nouvelobs", **kwargs)
            elif i==7:
                scrape_lesechos(outputfolder+"/lesechos", **kwargs)
            elif i==8:
                scrape_slate(outputfolder+"/slate", **kwargs)

        if args.loop:
            time.sleep(60*60*6) # wait 6 hours

        # Others TODO?
        # scrape_vice(outputfolder+"/vice")
        # scrape_lefigaro(outputfolder+"/lefigaro")
        # scrape_lepoint(outputfolder+"/lepoint")
        # scrape_liberation(outputfolder+"/liberation")
        # scrape_lejdd(outputfolder+"/lejdd")
        # scrape_lequipe(outputfolder+"/lequipe")
        # scrape_lci(outputfolder+"/lci")
        # scrape_ouestfrance(outputfolder+"/ouestfrance")
        # scrape_lesoir(outputfolder+"/lesoir")
        # scrape_ladepeche(outputfolder+"/ladepeche")
        # scrape_lavoixdunord(outputfolder+"/lavoixdunord")
        # scrape_lunion(outputfolder+"/lunion")
        # scrape_lanouvellerepublique(outputfolder+"/lanouvellerepublique")
        # scrape_lalsace(outputfolder+"/lalsace")
        #
        # ... Twitter
        # 
        # Aggrégateurs d'info:
        # https://news.google.com
        # https://comptoir.io/
        