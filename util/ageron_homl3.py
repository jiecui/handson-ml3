"""
Library of A Geron, Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 3rd Edition

Copyright 2021-2025 Richard J. Cui % Created: Mon 01/11/2021  3:21:04.437 PM;
Revision: 0.2  Date: Sat 01/11/2025 19:07:09.989634 PM

Rocky Creek Dr. NE
Rochester, MN 55906, USA

Email: richard.jie.cui@gmail.com
"""

import tarfile
import sys
import sklearn
import os
import matplotlib.pyplot as plt
import urllib.request
import pandas as pd

# check system requirements
# Python ≥ 3.5 is required
assert sys.version_info >= (3, 5)
# Scikit-Learn ≥ 0.20 is required
assert sklearn.__version__ >= "0.20"

# ==========================================================================
# Global
# ==========================================================================
# Create a function to save the figures.
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300,
             images_path='.'):
    '''Save the figure'''

    path = os.path.join(images_path, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# ==========================================================================
# Chapter 1
# ==========================================================================
# Chapter 1: Prepare country satisfaction data
def prepare_country_stats(oecd_bli, gdp_per_capita):
    '''PREPARE_COUNTRY_STATS function just merges the OECD's life
    satisfaction data and the IMF's GDP per capita data. It's a bit too long
    and boring and it's not specific to Machine Learning, which is why I
    left it out of the book.
    '''

    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"] == "TOT"]
    oecd_bli = oecd_bli.pivot(
        index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36))-set(remove_indices))
    sample_data = full_country_stats[[
        "GDP per capita", 'Life satisfaction']].iloc[keep_indices]
    missing_data = full_country_stats[[
        "GDP per capita", 'Life satisfaction']].iloc[remove_indices]
    return sample_data, missing_data


# Chapter 1: Download life satisfaction data
def download_lifesat(datapath='.'):
    '''Download life satisfaction data for Chapter 1

    We can get fresh data from the OECD's website and save it to
    ./datasets/lifesat/ Load and prepare GDP per capita data Just like
    above, you can update the GDP per capita data if you want. Just download
    data from http://goo.gl/j1MSKe (=> imf.org) and save it to
    datasets/lifesat/
    '''

    DOWNLOAD_ROOT = "http://raw.githubusercontent.com/ageron/handson-ml2/master/"
    os.makedirs(datapath, exist_ok=True)
    for filename in ("oecd_bli_2015.csv", "gdp_per_capita.csv"):
        print("Downloading", filename)
        url = DOWNLOAD_ROOT+"datasets/lifesat/"+filename
        urllib.request.urlretrieve(url, datapath+filename)
    return DOWNLOAD_ROOT

# ==========================================================================
# Chapter 2
# ==========================================================================
# Chapter 2: Download California housing data
def fetch_housing_data(datapath='.'):
    '''Dowload California housing data'''

    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
    os.makedirs(datapath, exist_ok=True)

    filename = "housing.tgz"
    print("Downloading", filename)
    url = DOWNLOAD_ROOT+"datasets/housing/"+filename
    urllib.request.urlretrieve(url, datapath+filename)

    # decompress the data
    housing_tgz=tarfile.open(datapath+filename)
    housing_tgz.extractall(path=datapath)
    housing_tgz.close()

# [EOF]
