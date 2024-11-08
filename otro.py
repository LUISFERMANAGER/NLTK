# Etiquetar un texto
import nltk
sentence = """At eight o'clock on Thursday morning... Arthur didn't feel very good."""
tokens = nltk.word_tokenize(sentence)
tokens
['At', 'eight', "o'clock", 'on', 'Thursday', 'morning',
'Arthur', 'did', "n't", 'feel', 'very', 'good', '.']
tagged = nltk.pos_tag(tokens)
tagged[0:6]
[('At', 'IN'), ('eight', 'CD'), ("o'clock", 'JJ'), ('on', 'IN'),
('Thursday', 'NNP'), ('morning', 'NN')]

# Identifica entidades nombradas
entities = nltk.chunk.ne_chunk(tagged)
entities 
('S', [('At', 'IN'), ('eight', 'CD'), ("o'clock", 'JJ'),
       ('on', 'IN'), ('Thursday', 'NNP'), ('morning', 'NN'),
       ('PERSON', [('Arthur', 'NNP')]),
       ('did', 'VBD'), ("n't", 'RB'), ('feel', 'VB'),
       ('very', 'RB'), ('good', 'JJ'), ('.', '.')])

# Mostrar el arbol de análisis
from nltk.corpus import treebank
t = treebank.parsed_sents('wsj_0001.mrg')[0]
t.draw()
