import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
import string

def create_unique_word_dict(text:list) -> dict:
    """
    A method that creates a dictionary where the keys are unique words
    and key values are indices
    """
    # Getting all the unique words from our text and sorting them alphabetically
    words = list(set(text))
    words.sort()

    # Creating the dictionary for the unique words
    unique_word_dict = {}
    for i, word in enumerate(words):
        unique_word_dict.update({
            word: i
        })

    return unique_word_dict    

def text_preprocessing(
    text:list,
    )->list:
    """
    A method to preproces text
    """
    punctuations = list(string.punctuation)
    stopWords = set(stopwords.words('english'))
    lancaster = LancasterStemmer()

    if not isinstance(text, str):
        return ""
    for x in text.lower(): 
        if x in punctuations: 
            text = text.replace(x, "")

    # Removing words that have numbers in them
    text = re.sub(r'\w*\d\w*', '', text)

    # Removing digits
    text = re.sub(r'[0-9]+', '', text)

    # Cleaning the whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Setting every word to lower
    text = text.lower()

    # Converting all our text to a list 
    text = text.split(' ')

    # Droping empty strings
    text = [x for x in text if x!='']

    # Droping stop words
    text = [x for x in text if x not in stopWords]

    # Stemming
    text = [lancaster.stem(x) for x in text]

    # Dropping synonyms 
    mary_synonyms = ['mother', 'virgin', 'madonn', 'immaculate', 'lady', 'queen', 'mariam', 'maria', 'marys']
    text = [x if x not in mary_synonyms else 'mary' for x in text]

    christ_syn = ['baby', 'babies', 'child', 'babys']
    text = [x if x not in christ_syn else 'baby' for x in text]

    royalty_syn = ['throne', 'crown', 'king', 'queen', 'majesty']
    text = [x if x not in royalty_syn else 'crown' for x in text]

    

    return text

# Functions to find the most similar word 
def euclidean(vec1:np.array, vec2:np.array) -> float:
    """
    A function to calculate the euclidean distance between two vectors
    """
    return np.sqrt(np.sum((vec1 - vec2)**2))

def find_similar(word:str, embedding_dict:dict, top_n=10)->list:
    """
    A method to find the most similar word based on the learnt embeddings
    """
    dist_dict = {}
    word_vector = embedding_dict.get(word, [])
    if len(word_vector) > 0:
        for key, value in embedding_dict.items():
            if key!=word:
                dist = euclidean(word_vector, value)
                dist_dict.update({
                    key: dist
                })

        return sorted(dist_dict.items(), key=lambda x: x[1])[0:top_n]       