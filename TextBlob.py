TextBlob
TextBlob stands on the giant shoulders of NLTK and pattern, and plays nicely with both.

Noun phrase extraction
Part-of-speech tagging
Sentiment analysis
Classification (Naive Bayes, Decision Tree)
Language translation and detection powered by Google Translate
Tokenization (splitting text into words and sentences)
Word and phrase frequencies
Parsing
n-grams
Word inflection (pluralization and singularization) and lemmatization
Spelling correction
Add new models or languages through extensions
WordNet integration


$ pip install -U textblob
$ python -m textblob.download_corpora



from textblob import TextBlob

text = '''
The titular threat of The Blob has always struck me as the ultimate movie
monster: an insatiably hungry, amoeba-like mass able to penetrate
virtually any safeguard, capable of--as a doomed doctor chillingly
describes it--"assimilating flesh on contact.
Snide comparisons to gelatin be damned, it's a concept with the most
devastating of potential consequences, not unlike the grey goo scenario
proposed by technological theorists fearful of
artificial intelligence run rampant.
'''

blob = TextBlob(text)
blob.tags           # [('The', 'DT'), ('titular', 'JJ'),
                    #  ('threat', 'NN'), ('of', 'IN'), ...]

blob.noun_phrases   # WordList(['titular threat', 'blob',
                    #            'ultimate movie monster',
                    #            'amoeba-like mass', ...])

for sentence in blob.sentences:
    print(sentence.sentiment.polarity)
# 0.060
# -0.341

blob.translate(to="es")  # 'La amenaza titular de The Blob...'


The pattern.en module contains a fast part-of-speech tagger for English 
(identifies nouns, adjectives, verbs, etc. in a sentence), sentiment analysis, 
tools for English verb conjugation and noun singularization & pluralization, and a WordNet interface.
