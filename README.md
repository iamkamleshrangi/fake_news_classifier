# Fake News Classifier

### Table of Contents
1. [Goal](#goal)
2. [Description](#description)
3. [Getting Started](#getting_started)
	1. [Dependencies](#dependencies)
	2. [File Structure](#file_structure)
	3. [Installing](#installation)
	4. [Instructions](#instructions)
	
4. [Acknowledgement](#acknowledgement)
5. [License](#license)

<a name="goal"></a>
### Goal
This project is designed to showcase the data and ML capabilities to deliver the solution for real-world problems, In this project, we collected data label data from kaggle in two CSV, first contains legitimate news and other csv contains fake news. 

<a name="description"></a>
### Description
This is built on top of a Natural Language Processing (NLP) trained model, and training based on the various prelabed dataset on the the legitimate and fake news.

Projects includes 
1. Data pipeline to clean, balance, select the relavent entities.
2. Saved the output of the CSV files for further process.
3. Used processed data, to create ML model for the fake classification.
4. Save the model for the classficication.

<a name="getting_started"></a>
### Getting Started

<a name="dependencies"></a>
#### Dependencies
* Python 3.8
* Notebook: Jupyter Notebook
* Data analysis libraries: Pandas, Numpy
* Machine Learning libraries: Scikit-Learn
* Natural Language Processing libraries: NLTK

<a name="installation"></a>
### Installing
* Clone the repository.
    ```
    git clone git@github.com:iamkamleshrangi/fake_news_classifier.git
    ```
* Proper conda/virtualenv enviroment ready with python3+.
* Install the necessary libraries provided in requirements.txt file.
* Follow the instructions provided in the next section.

<a name="instructions"></a>
#### Instructions

Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL & ML pipeline that cleans data and stores in database
        `python classifier.py `
    - The will result in output will be a classifier.
    - You may need good machine to run the process.


<a name="acknowledgements"></a>
## Acknowledgements
* [Kaggle](https://Kaggle.com/) for proposing idea and data set for the 

<a name="license"></a>
## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
