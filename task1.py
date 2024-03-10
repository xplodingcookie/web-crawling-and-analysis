import pandas as pd
import json
from typing import Dict, List

import requests
import bs4
import urllib
from robots import process_robots, check_link_ok

import re
from urllib.parse import urljoin

# A simple page limit used to catch procedural errors.
SAFE_PAGE_LIMIT = 1000

# Task 1 - Get All Links (3 marks)
def task1(starting_links: List[str], json_filename: str) -> Dict[str, List[str]]:
    # Crawl each url in the starting_link list, and output
    # the links you find to a JSON file, with each starting
    # link as the key and the list of crawled links for the
    # value.
    # Implement Task 1 here

    # The output dictionary
    crawled_all = {}

    # Iterate through each starting link
    for starter_link in starting_links:

        # Set up the initial page to crawl
        seed_url = starter_link
        page = requests.get(starter_link)
        soup = bs4.BeautifulSoup(page.text,'html.parser')

        robot_rules = process_robots(page.text)

        visited = {}
        visited[seed_url] = True
        num_visited = 1

        links = soup.find_all('a')

        # Exclude the seed_url from the links that are to be visited
        link_split = starter_link.split('/')
        seed_link = soup.find_all('a', href=re.compile(f'\/{link_split[3]}\/{link_split[4]}'))
        to_visit_relative = [l for l in links if l not in seed_link and "href" in l.attrs]

        # Note: We are only crawling in the /samplewiki/ and /fullwiki/ sections of the wiki
        pattern = re.compile(f'/{link_split[3]}/')

        # Get all links with the pattern and append them to the to_visit list
        to_visit = []
        for link in to_visit_relative:
            href = link.get("href")
            if not check_link_ok(robot_rules, href):
                continue
            if href is not None and pattern.match(href):
                to_visit.append(urljoin(seed_url, href))
        
        # Go through the to_visit links
        while (to_visit):
            # Impose the SAFE_PAGE_LIMIT
            if num_visited == SAFE_PAGE_LIMIT:
                break
            
            # Set up first item in to_visit list and remove it from the to_visit list
            link = to_visit.pop(0)
            page = requests.get(link)
            soup = bs4.BeautifulSoup(page.text, 'html.parser')

            # Add link to list of visited
            visited[link] = True

            # Find all URLs in this link
            new_links = soup.find_all('a')

            # Go through the links found 
            for new_link in new_links:
                
                # Check if there is an URL
                if "href" not in new_link.attrs:
                    continue
                
                new_item = new_link['href']
                
                # Skip the link if wiki does not want us to visit it
                if not check_link_ok(robot_rules, new_item):
                    continue

                # See if its in /samplewiki/ or /fullwiki/ section of wiki
                if pattern.match(new_item):
                    new_url = urljoin(link, new_item)
                    # Add it to the to_visit list if it has never been visited/ it is going to be visited
                    if new_url not in visited and new_url not in to_visit:
                        to_visit.append(new_url)
            
            # Update pages visited; obey the SAFE_PAGE_LIMIT
            num_visited += 1

        # After all the links are crawled from the seed_url add it to the dictionary crawled_all
        crawled_all[seed_url] = sorted(list(visited.keys()))
    
    return json.dump(crawled_all, open(json_filename, "w"))

