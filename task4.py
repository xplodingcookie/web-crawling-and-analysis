"""
COMP20008 Semester 1
Assignment 1 Task 4
"""

import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict
from collections import defaultdict

import numpy as np


# Task 4 - Plotting the Most Common Words (2 Marks)
def task4(bow: pd.DataFrame, output_plot_filename: str) -> Dict[str, List[str]]:
    # The bow dataframe is the output of Task 3, it has 
    # three columns, link_url, words and seed_url. The 
    # output plot should show which words are most common
    # for each seed_url. The visualisation is your choice,
    # but you should make sure it makes sense for what it
    # is meant to be.
    # Implement Task 4 here

    # Find unique seed_urls
    found_seed_urls = set()
    for seed_url in bow["seed_url"]:
        found_seed_urls.add(seed_url)

    # Top 10 words and occurances and what seed_url they are from: this will help plot graph
    word_freq = {}

    # Iterate through the seeds found
    unique_seed_urls = sorted(list(found_seed_urls))
    seed_top_10 = {}

    for unique_seed in unique_seed_urls:
        # Set up word counter
        word_count = defaultdict(int)
        # Make dataframe with only the seed_urls that match unique_seed
        current_seed = bow.loc[bow["seed_url"] == unique_seed]
        word_list = current_seed["words"]
        # Iterate through the "words" and count the words 
        for row in word_list:
            words = row.split(" ")
            for word in words:
                word_count[word] += 1

        # Sort by descending frequency and sort alphabetically    
        word_count = sorted(word_count.items(), key=lambda x: (-x[1], x[0]))
        
        # Add top 10 words to the dictionary seed_top_10
        top_10_words = [word[0] for word in word_count[:10]]
        seed_top_10[unique_seed] = top_10_words

        # Append to word and occurances to list
        word_freq[unique_seed] = [word for word in word_count[:10]]
        
    # Make grouped bar chart

    # Obtain x and y values needed to make graph
    # Create x-axis labels
    top_10_combined = []
    for unique_seed in unique_seed_urls:
        for word in seed_top_10[unique_seed]:
            if word not in top_10_combined:
                top_10_combined.append(word)
    unique_words = tuple(top_10_combined)
    
    # Data for the graph; the y-axis measures
    data = {}
    # Going to use this value to help manage axis scale
    max_value = 0
    for unique_seed in unique_seed_urls:
        num_word = []
        for word in unique_words:
            if word in seed_top_10[unique_seed]:
                for item in word_freq[unique_seed]:
                    # Find it and then append its frequency
                    if word == item[0]:
                        num_word.append(item[1])
            else:
                num_word.append(0)
        data[unique_seed] = tuple(num_word)

        # Keep track of max y value
        if max(num_word) > max_value:
            max_value = max(num_word)
    
    
    # Referring to https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
    # on how to make a grouped bar chart
    
    # Set up label locations
    # Space the label locations evenly apart
    x = np.arange(len(unique_words))
    width = 0.4
    multiplier = 0

    # Initiate making the plot
    # Note: constrained will allow matplotlib to maintain the best scale so
    # axes will fit
    fig, ax = plt.subplots(layout = 'constrained')
    
    # Create graph "bars"
    for attribute, occurance in data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, occurance, width, label = attribute)
        ax.bar_label(rects, padding=3, fontsize="7")
        multiplier += 1

    # Customise the aesthetics of graph to include title, axis labels, etc.
    ax.set_ylabel("Frequency of words")
    ax.set_xlabel("Unique most common words from two seed URls")
    ax.set_title("Comparison of the top 10 words from two seed URLs")
    # Note: use width/2 to keep label of words in middle of the grouped bars
    ax.set_xticks(x + width/2, unique_words, rotation = 90)
    ax.legend(fontsize = "8")
    # Set y dimensions so legend does not block any of the data
    # Note: 0.23 is just an arbitrary scalar that made sure legend didn't block info
    ax.set_ylim([0, max_value + max_value * 0.23])
    
    # Output as png file
    plt.savefig(output_plot_filename)

    return seed_top_10
