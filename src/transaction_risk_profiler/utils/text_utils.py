import bs4


def extract_text(x):
    soup = bs4.BeautifulSoup(x)
    return soup.text
