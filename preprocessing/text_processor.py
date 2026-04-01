#normalize text, tokenize, remove stop words, etc.
class TextProcessor:
    def tokenize(self, text):
        return text.lower().split() 