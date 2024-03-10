import json

import requests
import bs4
import urllib
import unicodedata
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from robots import process_robots, check_link_ok

from bs4 import BeautifulSoup
import nltk

# Task 2 - Extracting Words from a Page (4 Marks)
def task2(link_to_extract: str, json_filename: str):
    # Download the link_to_extract's page, process it 
    # according to the specified steps and output it to
    # a file with the specified name, where the only key
    # is the link_to_extract, and its value is the 
    # list of words produced by the processing.
    # Implement Task 2 here

    # Set up page to extract words from
    link = link_to_extract
    page = requests.get(link)
    page.encoding = page.apparent_encoding
    soup = BeautifulSoup(page.text, 'html.parser')

    section = soup.find('div', id="mw-content-text")

    # First 8 steps of narrowing down and removing all irrelevant elements from page
    # Remove all th elements with class of infobox-label
    for th in section.find_all('th', class_="infobox-label"):
        th.decompose()

    # Remove all div elements with class of printfooter
    for div in section.find_all('div', class_="printfooter"):
        div.decompose()

    # Remove all div elements with id of toc
    for div in section.find_all('div', id="toc"):
        div.decompose()

    # Remove all table elements with class of ambox
    for table in section.find_all('table', class_="ambox"):
        table.decompose()

    # Remove all div elements with class of asbox
    for div in section.find_all('div', class_="asbox"):
        div.decompose()

    # Remove all span elements with class of mw-editsection
    for span in section.find_all('span', class_="mw-editsection"):
        span.decompose()
    
    narrowed_section = section.get_text(' ')

    # Second section produce word tokens and remove irrelevant tokens
    # Casefold and normalize all page text into NFKD form
    narrowed_section = unicodedata.normalize("NFKD", narrowed_section.casefold())

    # Anything not in a-z, \s and '\'
    narrowed_section = re.sub(r"[^a-z\s\\]", ' ', narrowed_section)

    # Convert all spacing characters (\s) into a single space (' ')
    # Note: the + is for the case of something like '\t\t'
    narrowed_section = re.sub("\s+", ' ', narrowed_section)
    
    # Convert text to explicit tokens, ' ' is the boundary between tokens
    # Note: specifies single-space boundary so I have decided to use .split(' ')
    # instead of explict_tokens = nltk.word_tokenize(narrowed_section) also I found
    # "cannot" would cause issues with the check of task3
    explict_tokens = narrowed_section.split(' ')

    # Set up set of stopwords and filter them out
    stop_words = set(stopwords.words('english'))
    no_stopwords = [word for word in explict_tokens if not word in stop_words]

    # No stopwords and words less than two character lengths
    updated_no_stopwords = [word for word in no_stopwords if len(word) >= 2]

    # Use porterstemmer to find stems of all the words
    porterStemmer = PorterStemmer()
    stemmed = [porterStemmer.stem(word) for word in updated_no_stopwords]

    # Set up dictionary to be outputted
    extracted_words = {}
    extracted_words[link_to_extract] = stemmed

    return json.dump(extracted_words, open(json_filename, "w"))
