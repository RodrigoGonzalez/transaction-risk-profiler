from collections import Counter

import bs4
import pandas as pd
from bs4 import BeautifulSoup
from nltk import PorterStemmer
from nltk.corpus import stopwords

STEMMER = PorterStemmer()


def extract_content_text(content: bs4.element.Tag) -> str:
    """
    Extracts text from a BeautifulSoup Tag object if it has a text attribute.

    Parameters
    ----------
    content : bs4.element.Tag
        A BeautifulSoup Tag object.

    Returns
    -------
    str
        The text attribute of the Tag object if it exists, else an empty
        string.
    """
    return content.text if hasattr(content, "text") else ""


def clean_description(text: str) -> str:
    """
    Cleans the input text by removing non-breaking space characters, extracting
    text from HTML tags, removing English stopwords, and applying stemming.

    Parameters
    ----------
    text : str
        The input text to be cleaned.

    Returns
    -------
    str
        The cleaned text.
    """
    text = text.replace("\xa0", " ")
    soup = BeautifulSoup(text, "html.parser")
    text = " ".join([extract_content_text(x) for x in soup.contents])
    stop_words = stopwords.words("english")
    text = [x for x in text.split() if x not in stop_words]
    return " ".join([STEMMER.stem(x) for x in text])


def count_links(text: str) -> int:
    """
    Counts the number of hyperlinks in the input text.

    Parameters
    ----------
    text : str
        The input text.

    Returns
    -------
    int
        The number of hyperlinks in the text.
    """
    text = text.replace("\xa0", " ")
    soup = BeautifulSoup(text, "html.parser")
    links = [a.attrs["href"] for a in soup.find_all("a") if a.has_attr("href")]
    return len(links)


def extract_font_and_link_count(text: str) -> pd.Series:
    """
    Extracts the most common font family and the number of hyperlinks from the
    input text.

    Parameters
    ----------
    text : str
        The input text.

    Returns
    -------
    pd.Series
        A Pandas' Series object with the most common font family and the number
        of hyperlinks.
    """
    text = text.replace("\xa0", " ")
    soup = BeautifulSoup(text, "html.parser")

    try:
        counter = Counter([text.attrs["style"] for text in soup.find_all("span")])
        font_family = counter.most_common()[0][0]
    except IndexError:
        font_family = ""

    links = [a.attrs["href"] for a in soup.find_all("a") if a.has_attr("href")]
    return pd.Series({"font": font_family, "link_count": len(links)})


def extract_text(text: str) -> str:
    """
    Extracts the text from the input HTML string.

    Parameters
    ----------
    text : str
        The input HTML string.

    Returns
    -------
    str
        The extracted text.
    """
    return bs4.BeautifulSoup(text).text
