from nltk.corpus import stopwords
 
french_stopwords = set(stopwords.words("french"))
new_stopwords = {
    "s'",
    "quel",
    "que",
    "quoi",
    "comment",
    "l'",
    "d'",
    "mais",
    "ou",
    "et",
    "donc",
    "or",
    "ni",
    "car",
    "quelle",
    "quelles",
    "pourquoi",
}
french_stopwords.update(new_stopwords)

def remove_french_stopwords(text):
    text = text.lower()
    tokens = text.split()  # Split text into words
    filtered_tokens = [token for token in tokens if token.lower() not in french_stopwords]
    return " ".join(filtered_tokens)