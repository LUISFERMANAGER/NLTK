import nltk 
import random 

data = [
    ("Me gusta esta película", "positivo"),
    ("Esta película es pesima", "negativo"),
    ("Esta película es genial", "positivo"),
    ("No soporto ver esta película", "negativo"),
    ("La actuación en esta película es un fenómeno", "positivo"),
    ("Lamento haber perdido el tiempo con estas películas", "negativo"),
    ("Disfruté muchísimo viendo esta película", "positivo"),
    ("A esta película le falta profundidad y sustancia", "negativo"),
    ("La trama de esta película fue cautivadora", "positivo"),
    ("Los personajes de esta película me parecieron muy atractivos", "positivo"),
    ("Los efectos especiales de esta película fueron impresionantes", "positivo"),
    ("La trama era precisa y poco original", "negativo"),
    ("Me decepcionó la falta de desarrollo del carácter", "negativo"),
    ("La cinematografía de esta película fue impresionante", "positivo"),
    ("El diálogo se sintió forzado y antinatural", "negativo"),
    ("El ritmo de la película fue demasiado lento para mi gusto", "negativo"),
    ("Me sorprendió gratamente lo mucho que disfruté esta película", "positivo"),
    ("El final me dejó insatisfecho y confundido", "negativo"),
    ("La película superó mis expectativas", "positivo"),
    ("La actuación de los actores fue mediocre", "negativo")
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
print("Exactitud:", accuracy)


# Clasificamos un nuevo texto
new_text = "This movie is amazing"
new_text_features = preprocess(new_text)
predicted_label = classifier.classify(new_text_features)
print("Predicción de sentimiento es:", predicted_label)