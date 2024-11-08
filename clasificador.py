import nltk 
import random 

data = [
    ("I love this movie", "positive"),
    ("This movie is terrible", "negative"),
    ("This movie is great", "positive"),
    ("I can't stand watching this movie", "negative"),
    ("The acting in this movie is phenomena", "positive"),
    ("I regret wasting my time on this films", "negative"),
    ("I thoroughly enjoyed watching this movie", "positive"),
    ("This movie lacks depth an substance", "negative"),
    ("The plot of this movie was captivating", "positive"),
    ("I found the characters in this film to be very engaging", "positive"),
    ("The special effects in this movie were impressive", "positive"),
    ("The storyline was precitable and unoriginal", "negative"),
    ("I was disappointed by the lack of character delopment", "negative"),
    ("The cinematography in this film was stunning", "positive"),
    ("The dialogue felt forced and unnatural ", "negative"),
    ("The pacing of the movie was too slow for my liking", "negative"),
    ("I was plesantly surprised by how much I enjoyed this fiml", "positive"),
    ("The ending left me feeling unsatisfied and confused", "negative"),
    ("The movie exceeded my expectations", "positive"),
    ("The performance by the actors were lackluter", "negative")
]
# Preprocesamiento de datos: Tokenización y extracción de características

def preprocess(text):
    tokens = nltk.word_tokenize(text)
    return {word: True for word in tokens}

# Aplicamos el preprocesamiento a los datos
featuresets = [(preprocess(text), sentiment) for text, sentiment in data]

# Dividimos los datos en conjuntos de entrenamiento y prueba
train_set, test_set = featuresets[:16], featuresets[16:]

# Entrenamos un clasificador utilizando Naive Bayes
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Evaluamos el clasificador en el conjunto de prueba

accuracy = nltk.classify.accuracy(classifier, test_set)
print("Accuracy:", accuracy)


# Clasificamos un nuevo texto
new_text = "This movie is amazing"
new_text_features = preprocess(new_text)
predicted_label = classifier.classify(new_text_features)
print("Predicted sentiment:", predicted_label)