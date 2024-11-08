import nltk
from nltk.stem import PorterStemmer
word = ["running", "plays", "jumped"]
stemmer = PorterStemmer()
stems = [stemmer.stem(word) for word in word]
print(stems)
