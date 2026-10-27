import random
import re
from collections import Counter, defaultdict


class BigramModel:
    """Simple bigram language model built from a list of text strings."""

    def __init__(self, corpus):
        # bigram_counts[w1][w2] = number of times w2 follows w1
        self.bigram_counts = defaultdict(Counter)
        self.fit(corpus)

    @staticmethod
    def _tokenize(text):
        return re.findall(r"[\w']+", text.lower())

    def fit(self, corpus):
        for sentence in corpus:
            tokens = self._tokenize(sentence)
            for w1, w2 in zip(tokens, tokens[1:]):
                self.bigram_counts[w1][w2] += 1

    def generate_text(self, start_word, length):
        """Generate up to `length` words, starting from `start_word`."""
        word = start_word.lower()
        output = [word]
        for _ in range(max(length, 1) - 1):
            followers = self.bigram_counts.get(word)
            if not followers:
                break  # dead end: no known word follows this one
            words, counts = zip(*followers.items())
            word = random.choices(words, weights=counts, k=1)[0]
            output.append(word)
        return " ".join(output)
