# Image Caption Generation using Deep Learning

[![GitHub license](https://img.shields.io/github/license/your-username/your-repo)](https://github.com/Sajid030/image-captioning/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/-TensorFlow-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?logo=jupyter&logoColor=white)
[![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B)](https://www.streamlit.io/)

## Table of Contents

- [Demo](#demo)
- [Overview](#overview)
- [About the Dataset](#about-the-dataset)
- [Installation](#installation)
- [Deployement on Streamlit](#deployement-on-streamlit)
- [Directory Tree](#directory-tree)
- [Bug / Feature Request](#bug--feature-request)
- [Future Scope](#future-scope)

## Demo

- Link: 

`Note:` If the website link provided above is not working, it might mean that the deployment has been stopped or there are technical issues. We apologize for any inconvenience.

- Please consider giving a ⭐ to the repository if you find this app useful.
- A quick preview of the **Image Caption Generator** app:

![Caption Generator Demo](demo.gif)

## Overview

This repository contains code for an image caption generation system using deep learning techniques. The system leverages a pretrained VGG16 model for feature extraction and a custom captioning model which was trained using LSTM for generating captions. The model is trained on the Flickr8k dataset using an attention mechanism to improve caption quality.

The key components of the project include:

- Image feature extraction using a pretrained VGG16 model
- Caption preprocessing and tokenization
- Custom captioning model architecture with attention mechanism
- Model training and evaluation
- Streamlit app for interactive caption generation

## About the Dataset

The [Flickr8k dataset](https://www.kaggle.com/adityajn105/flickr8k) is used for training and evaluating the image captioning system. It consists of 8,091 images, each with five captions describing the content of the image. The dataset provides a diverse set of images with multiple captions per image, making it suitable for training caption generation models.

Download the dataset from [Kaggle](https://www.kaggle.com/adityajn105/flickr8k) and organize the files as follows:

- flickr8k
  - Images
    - (image files)
  - captions.txt

## Installation

This project is written in Python 3.11.4. If you don't have Python installed, you can download it from the [official website](https://www.python.org/downloads/). If you have an older version of Python, you can upgrade it using the pip package manager, which should be already installed if you have Python 2 >=2.7.9 or Python 3 >=3.4 on your system.
To install the required packages and libraries, you can use pip and the provided requirements.txt file. First, clone this repository to your local machine using the following command:
```
https://github.com/Sajid030/image-captioning.git
```
Once you have cloned the repository, navigate to the project directory and run the following command in your terminal or command prompt:
```bash
pip install -r requirements.txt
```
This will install all the necessary packages and libraries needed to run the project.

## Deployement on Streamlit

1. Create an account on Streamlit Sharing.
2. Fork this repository to your GitHub account.
3. Log in to Streamlit Sharing and create a new app.
4. Connect your GitHub account to Streamlit Sharing and select this repository.
5. Set the following configuration variables in the Streamlit Sharing dashboard:
```
[server]
headless = true
port = $PORT
enableCORS = false
```
6. Click on "Deploy app" to deploy the app on Streamlit Sharing.

## Directory Tree

```
├── model
│   ├── dataset.pkl
│   ├── my_model.h5
├── resource 
│   ├── loanprediction.gif
├── app.py
├── LICENSE.md
├── loan_prediction.ipynb
├── README.md
├── requirements.txt
```

## Bug / Feature Request

If you encounter any bugs or issues with the loan status predictor app, please let me know by opening an issue on my [GitHub repository](https://github.com/Sajid030/image-captioning/issues). Be sure to include the details of your query and the expected results. Your feedback is valuable in helping me improve the app for all users. Thank you for your support!

## Future Scope

1. **Fine-tuning**: Experiment with fine-tuning the captioning model architecture and hyperparameters for improved performance.

2. **Dataset Expansion**: Incorporate additional datasets to increase the diversity and complexity of the trained model for example we can train the model on [Flickr30k dataset](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset).

3. **Beam Search**: Implement beam search decoding for generating multiple captions and selecting the best one.

4. **User Interface Enhancements**: Improve the Streamlit app's user interface and add features such as image previews and caption confidence scores.

5. **Multilingual Captioning**: Extend the model to generate captions in multiple languages by incorporating multilingual datasets.