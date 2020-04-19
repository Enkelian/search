from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string


class TextPreprocessor:
    def __init__(self):
        self.ps = PorterStemmer()

    def clear_text(self, text):
        text = "".join([c for c in text if c not in string.punctuation])
        return text.lower()

    def text_to_words(self, text):
        text = self.clear_text(text)
        words = word_tokenize(text)
        words = self.remove_stop_words(words)
        return words

    def is_ascii(self, s):
        return all(ord(c) < 128 for c in s)

    def remove_stop_words(self, words):
        stop_words = set(stopwords.words('english'))
        words = [self.ps.stem(w) for w in words if (w not in stop_words and self.is_ascii(w))]
        return words

