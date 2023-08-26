from collections import Counter

import bs4
import pandas as pd
from bs4 import BeautifulSoup
from nltk import PorterStemmer
from nltk.corpus import stopwords

STEMMER = PorterStemmer()


def try_text(content):
    return content.text if hasattr(content, "text") else ""


def clean_description(text):
    text = text.replace("\xa0", " ")
    soup = BeautifulSoup(text, "html.parser")
    text = " ".join([try_text(x) for x in soup.contents])
    stop_words = stopwords.words("english")
    text = [x for x in text.split() if x not in stop_words]
    return " ".join([STEMMER.stem(x) for x in text])


def feature_descriptions(text):
    text = text.replace("\xa0", " ")
    soup = BeautifulSoup(text, "html.parser")

    links = [a.attrs["href"] for a in soup.find_all("a") if a.has_attr("href")]
    return len(links)


def feature_descriptions_by_fonts(text):
    text = text.replace("\xa0", " ")
    soup = BeautifulSoup(text, "html.parser")

    try:
        counter = Counter([text.attrs["style"] for text in soup.find_all("span")])
        font_family = counter.most_common()[0][0]

    except IndexError:
        font_family = ""

    links = [a.attrs["href"] for a in soup.find_all("a") if a.has_attr("href")]
    return pd.Series({"font": font_family, "link_count": len(links)})


def extract_text(a):
    return bs4.BeautifulSoup(a).text
