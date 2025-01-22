"""
Library of A Geron, Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 3rd Edition

Copyright 2021-2025 Richard J. Cui Created: Mon 01/11/2021  3:21:04.437 PM
Revision: 0.5  Date: Wed 01/22/2025 03:26:42.811992 PM

Rocky Creek Dr. NE
Rochester, MN 55906, USA

Email: richard.jie.cui@gmail.com
"""

# ==========================================================================
# Constants
# ==========================================================================
from sklearn.datasets import fetch_openml
import joblib
import pandas as pd
import urllib.request
import matplotlib.pyplot as plt
import shutil
import os
import sklearn
import sys
import tarfile
import PIL
DOWNLOAD_ROOT_OLD = "http://raw.githubusercontent.com/ageron/handson-ml2/master/"  # 2nd Edition
DOWNLOAD_ROOT = "https://github.com/ageron/data/raw/main/"  # 3rd Edition
HOML3_ROOT = "https://github.com/ageron/handson-ml3/raw/main/"

# ==========================================================================
# Libraries
# ==========================================================================

# check system requirements
# Python ≥ 3.5 is required
assert sys.version_info >= (3, 5)
# Scikit-Learn ≥ 0.20 is required
assert sklearn.__version__ >= "0.20"

# ==========================================================================
# Global
# ==========================================================================
# get model root path


def get_model_root():
    '''Get model root path'''

    models_path = os.path.join(os.path.dirname(__file__), "..", "..", "models")
    return os.path.abspath(models_path)

# get image root path


def get_image_root():
    '''Get image root path'''

    images_path = os.path.join(os.path.dirname(__file__), "..", "..", "images")
    return os.path.abspath(images_path)

# get data root path


def get_data_root():
    '''Get data root path'''

    data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    return os.path.abspath(data_path)

# Create a function to save the figures.


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300,
             images_folder="images"):
    '''Save the figure'''

    images_path = os.path.join(get_image_root(), images_folder)
    os.makedirs(images_path, exist_ok=True)
    path = os.path.join(images_path, f"{fig_id}.{fig_extension}")
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

# Chapter 1: Load life satisfaction data


def load_lifesat():
    '''Load life satisfaction data for Chapter 1'''

    data_root = get_data_root()
    csv_path = os.path.join(data_root, "lifesat", "lifesat.csv")
    return pd.read_csv(csv_path)

# Chapter 1: Download life satisfaction data


def download_lifesat():
    '''Download life satisfaction data for Chapter 1

    We can get fresh data from the OECD's website and save it to
    ./datasets/lifesat/ Load and prepare GDP per capita data Just like
    above, you can update the GDP per capita data if you want. Just download
    data from http://goo.gl/j1MSKe (=> imf.org) and save it to
    datasets/lifesat/
    '''

    datapath = os.path.join(get_data_root(), "lifesat")
    os.makedirs(datapath, exist_ok=True)

    for filename in ("oecd_bli_2015.csv", "gdp_per_capita.csv"):
        print("Downloading", filename)
        url = DOWNLOAD_ROOT_OLD+"datasets/lifesat/"+filename
        urllib.request.urlretrieve(url, os.path.join(datapath, filename))

    filename = 'lifesat.csv'
    print("Downloading", filename)
    url = DOWNLOAD_ROOT + "lifesat/" + filename
    urllib.request.urlretrieve(url, os.path.join(datapath, filename))

# ==========================================================================
# Chapter 2
# ==========================================================================
# Chapter 2: Load California housing data


def load_housing_data():
    '''Load California housing data'''

    data_root = get_data_root()
    csv_path = os.path.join(data_root, "housing", "housing.csv")
    return pd.read_csv(csv_path)

# Chapter 2: Download California housing data


def download_housing_data():
    '''Dowload California housing data'''

    data_root = get_data_root()
    datapath = os.path.join(data_root, "housing")
    os.makedirs(datapath, exist_ok=True)

    filename = "housing.tgz"
    print("Downloading", filename)
    tarball_path = os.path.join(data_root, filename)
    url = DOWNLOAD_ROOT+filename
    urllib.request.urlretrieve(url, tarball_path)

    # decompress the data
    with tarfile.open(tarball_path) as housing_tgz:
        housing_tgz.extractall(path=data_root)
    # move the extracted file to the datapath, overwrite it if already exists
    try:
        destination_file = os.path.join(datapath, filename)
        if os.path.exists(destination_file):
            print(f"Removing existing file '{destination_file}'")
            os.remove(destination_file)
        shutil.move(tarball_path, datapath)
        print(f"File '{tarball_path}' moved successfully to '{datapath}'.")
    except Exception as e:
        print(f"Error moving file: {e}")

# Chapter 2: Read California image


def read_california_image():
    '''Read California image'''

    image_root = get_image_root()
    imagepath = os.path.join(image_root, "end_to_end_project")
    filename = "california.png"
    image_file = os.path.join(imagepath, filename)

    # download the image if it doesn't exist
    if not os.path.exists(image_file):
        download_california_image()

    return plt.imread(image_file)

# Chapter 2: Download California image


def download_california_image():
    '''Download California image'''

    image_root = get_image_root()
    imagepath = os.path.join(image_root, "end_to_end_project")
    os.makedirs(imagepath, exist_ok=True)

    filename = "california.png"
    print("Downloading", filename)
    url = HOML3_ROOT+"images/end_to_end_project/"+filename
    urllib.request.urlretrieve(url, os.path.join(imagepath, filename))

# ==========================================================================
# Chapter 3
# ==========================================================================
# Chapter 3: Load MNIST data


def load_mnist_data():
    '''Load MNIST data'''

    data_root = get_data_root()
    datapath = os.path.join(data_root, "mnist")
    filename = "mnist_784"
    mnist_file = os.path.join(datapath, f"{filename}.joblib")

    # download the data if it doesn't exist
    if not os.path.exists(mnist_file):
        download_mnist_data()

    return joblib.load(mnist_file)

# Chapter 3: Download MNIST data


def download_mnist_data():
    '''Download MNIST data'''

    data_root = get_data_root()
    datapath = os.path.join(data_root, "mnist")
    os.makedirs(datapath, exist_ok=True)

    filename = "mnist_784"
    print("Downloading", filename)
    minst_file = os.path.join(datapath, f"{filename}.joblib")
    mnist = fetch_openml(filename, version=1, as_frame=False, parser='pandas')
    joblib.dump(mnist, minst_file)

# ==========================================================================
# Chapter 9
# ==========================================================================


def load_ladybug_image():
    '''Load ladybug image'''

    image_root = get_image_root()
    imagepath = os.path.join(image_root, "unsupervised_learning")
    filename = "ladybug.png"
    image_file = os.path.join(imagepath, filename)

    # download the image if it doesn't exist
    if not os.path.exists(image_file):
        download_ladybug_image()

    return PIL.Image.open(image_file)


def download_ladybug_image():
    '''Download ladybug image'''

    image_root = get_image_root()
    imagepath = os.path.join(image_root, "unsupervised_learning")
    os.makedirs(imagepath, exist_ok=True)

    filename = "ladybug.png"
    print("Downloading", filename)
    url = HOML3_ROOT+"images/unsupervised_learning/"+filename
    urllib.request.urlretrieve(url, os.path.join(imagepath, filename))

# [EOF]
