"""
COMP20008 Semester 1
Assignment 1 Task 5
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Union, List

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer

# Task 5 - Dimensionality Reduction (3 marks)
def task5(bow_df: pd.DataFrame, tokens_plot_filename: str, distribution_plot_filename: str) -> Dict[str, Union[List[str], List[float]]]:
    # bow_df is the output of Task 3, for this task you 
    # should generate a bag of words, normalisation of the 
    # data perform PCA decomposition to 2 components, and 
    # then plot all URLs in a way which helps you answer
    # the discussion questions. If you would like to verify 
    # your PCA results against the sample data, you can return
    # the PCA weights - containing the list of most positive
    # weighted words, most negatively weighted words and the 
    # weights in the PCA decomposition for each respective word.
    # Implement Task 5 here

    # Initialize CountVectorizer to get a BoW representation of the words
    vectorizer = CountVectorizer()
    bow = vectorizer.fit_transform(bow_df["words"])
    bow_vector_df = pd.DataFrame(bow.toarray())

    # Perform normalisation, use sklearn's norm="max"
    normalize_bow_values = Normalizer(norm="max").transform(bow_vector_df)
    normalize_df = pd.DataFrame(normalize_bow_values)

    # Get the names of the words
    feature_names = vectorizer.get_feature_names_out()

    # Perform PCA with 2 components to the dataframe
    pca_decomp = PCA(n_components=2, random_state=535)
    bow_pca = pca_decomp.fit_transform(normalize_df)

    # Create the pca weight dictionary
    pca_weight_dict = {}
    for feature in [0, 1]:
        # Create a "zip" of the word and its weight
        word_weight = zip(pca_decomp.components_[feature], feature_names)

        # Create the nested dictionary with pos, neg, pos_weights and negative_weights
        sorted_items = sorted(word_weight, reverse=True)
        top_10_token = {}
        top_10_token["positive"] = [item[1] for item in sorted_items[:10]]
        top_10_token["negative"] = [item[1] for item in sorted_items[-10:]]
        top_10_token["positive_weights"] = [item[0] for item in sorted_items[:10]]
        top_10_token["negative_weights"] = [item[0] for item in sorted_items[-10:]]
        
        # Append to dictionary
        pca_weight_dict[str(feature)] = top_10_token
    
    # Make grouped bar chart for plot of tokens and their weights
    # Create the x-axis labels
    pos_neg_word_unique = []
    for component in ["0", "1"]:
        for sign in ["positive", "negative"]:
            for word in pca_weight_dict[component][sign]:
                if word not in pos_neg_word_unique:
                    pos_neg_word_unique.append(word)
    unique_words = tuple(pos_neg_word_unique)

    # Create data dictionary for grouped bar graph; measures of the y value
    # ; weight
    data = {}

    for component in ["0", "1"]:
        word_weight_values = []
        pos_neg_words = pca_weight_dict[component]["positive"] \
                        + pca_weight_dict[component]["negative"]
        pos_neg_weights = pca_weight_dict[component]["positive_weights"] \
                        + pca_weight_dict[component]["negative_weights"]
        word_weight = list(zip(pos_neg_weights, pos_neg_words))
        for word in unique_words:
            if word in pos_neg_words:
                for item in word_weight:
                    if word == item[1]:
                        word_weight_values.append(item[0])
            else:
                word_weight_values.append(0)
        data[f'Component {component}'] = tuple(word_weight_values)

    # Set up label locations
    x = np.arange(len(unique_words))
    width = 0.4
    multiplier = 0

    fig, ax = plt.subplots(layout = 'constrained')

    # Create the graph "bars"
    for component, weight in data.items():
        offset = width * multiplier
        ax.bar(x + offset, weight, width, label = component)
        multiplier += 1
    
    # Customise the aesthetics of the graph to include title, axis labels, etc.
    ax.set_ylabel("Weight of token")
    ax.set_xlabel("Unique most strongly weighted words from PCA")
    ax.set_title("Comparison of tokens that most heavily affected PCA components")
    # Note: use width/2 to keep label of words in middle of the grouped bars
    ax.set_xticks(x + width/2, unique_words, rotation = 90)
    ax.legend(fontsize = "8")
    ax.axhline(linewidth=0.5, color='k')
    
    plt.savefig(tokens_plot_filename)

    # Clear plot to make second plot
    plt.clf()

    # Create scatterplot for distribution of PCA

    # Get the seed_url names
    found_seeds = set()
    for seed_url in bow_df["seed_url"]:
        found_seeds.add(seed_url)

    unique_seed_urls = sorted(list(found_seeds))

    # find the indexes that match the words to seed URL
    seed_url_1_idx = list(bow_df.loc[bow_df["seed_url"] == unique_seed_urls[0]].index)
    seed_url_2_idx = list(bow_df.loc[bow_df["seed_url"] == unique_seed_urls[1]].index)
    
    # Create the two dataframes to scatter
    # Note: Setting index column name to the seed_url so it can be easily accessed 
    # while plotting scatterplot; easy to create legend and set colours if needed 
    pca_seed_url_1 = pd.DataFrame([bow_pca[i] for i in seed_url_1_idx], columns=["pca_0", "pca_1"])
    pca_seed_url_1.index.name = unique_seed_urls[0]
    pca_seed_url_2 = pd.DataFrame([bow_pca[i] for i in seed_url_2_idx], columns=["pca_0", "pca_1"])
    pca_seed_url_2.index.name = unique_seed_urls[1]

    # Create the overlapping scatterplot
    for seed_url in [pca_seed_url_1, pca_seed_url_2]:
        plt.scatter(x=seed_url["pca_0"], y=seed_url["pca_1"], label=seed_url.index.name)
    
    # Customise the aesthetics of graph to include title, axis labels, etc.
    plt.title("PCA applied to Bag of Words from 2 seed URLs")
    plt.grid()
    plt.xlabel("1st Principal Component")
    plt.ylabel("2nd Principal Component")
    plt.legend(fontsize="8")

    # Save png to second filename given
    plt.savefig(distribution_plot_filename)

    return pca_weight_dict
