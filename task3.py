""" 
COMP20008 Semester 1
Assignment 1 Task 3
"""

from typing import Dict, List
import pandas as pd
import json
import requests
import bs4
import urllib
import unicodedata
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from robots import process_robots, check_link_ok

# just importing BeautifulSoup so no need to do bs4.BeautifulSoup
from bs4 import BeautifulSoup
import nltk

# Task 3 - Producing a Bag Of Words for All Pages (2 Marks)
def task3(link_dictionary: Dict[str, List[str]], csv_filename: str):
    # link_dictionary is the output of Task 1, it is a dictionary
    # where each key is the starting link which was used as the 
    # seed URL, the list of strings in each value are the links 
    # crawled by the system. The output should be a csv which
    # has the link_url, the words produced by the processing and
    # the seed_url it was crawled from, this should be output to
    # the file with the name csv_filename, and should have no extra
    # numeric index.
    # Implement Task 3 here

    # Link is the key, seed_url corresponding to link and words corresponding to 
    # link are the values
    link_seedurl_dict = {}
    link_words_dict = {}
    
    # Go through each link and extract the words using function from task2
    # and append to dictionaries
    for seed_url in link_dictionary:

        for link_url in link_dictionary[seed_url]:
            
            link_seedurl_dict[link_url] = seed_url
            
            link_words_dict[link_url] = extract_words(link_url)

    # Turn dictionaries into dataframe
    link_words = pd.Series(link_words_dict)
    link_seedurl = pd.Series(link_seedurl_dict)

    BoW_dataframe = pd.DataFrame({"words": link_words, "seed_url": link_seedurl})

    # Change the index into the first column and rename index to link_url
    BoW_dataframe = BoW_dataframe.reset_index().rename(columns={"index": "link_url"})

    # Now we sort the dataframe in ascending order by link_url and then seed_url
    BoW_dataframe = BoW_dataframe.sort_values(["link_url", "seed_url"], ascending=True)

    # Return csv file of the dataframe
    return BoW_dataframe.to_csv(csv_filename, index=False)

def extract_words(link_to_extract):

    """ takes a link and extracts all of the words according to the steps given in task2
    and returns a string with ' ' between all of the words """
    
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

    return ' '.join(stemmed)
