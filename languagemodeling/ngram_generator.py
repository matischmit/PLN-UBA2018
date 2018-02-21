from collections import defaultdict
import random


class NGramGenerator(object):

    def __init__(self, model):
        """
        model -- n-gram model.
        """
        self._n = model._n
        self._probs = defaultdict(dict)
        tokens = list(model._count.keys())
        ngrams = list(filter(lambda i: len(i) is self._n, tokens)) #uso solo tokens del tamaño del ngrama
        for tokens in ngrams:

            token = tokens[-1]
            prev_tokens = tokens[:-1]
            probability = model.cond_prob(token, list(prev_tokens))

            if prev_tokens not in self._probs:
                self._probs[prev_tokens] = dict()

            self._probs[prev_tokens][token] = probability

        self._sorted_probs = {}
        for token_probs in self._probs:
            self._sorted_probs[token_probs] = sorted(self._probs[token_probs].items())

    def generate_sent(self):
        """Randomly generate a sentence."""
        n = self._n

        sent = []
        prev_tokens = ['<s>'] * (n - 1)

        token = self.generate_token(tuple(prev_tokens))
        while token != '</s>':
            sent.append(token)              #agrego nuevo
            prev_tokens.append(token)       #agrego el nuevo a los ya consumidos
            prev_tokens = prev_tokens[1:]    #para mantener el tamaño del ngrama descarto el primero
            token = self.generate_token(tuple(prev_tokens)) #recalculo nueva palabra
        return sent

    def generate_token(self, prev_tokens=None):
        """Randomly generate a token, given prev_tokens.

        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        n = self._n
        if not prev_tokens:
            prev_tokens = ()
        assert len(prev_tokens) == n - 1

        r = random.random()
        probs = self._sorted_probs[prev_tokens]

        i = 0
        token, prob = probs[0]
        acum = prob
        while r > acum and i < len(probs) - 1:
            i += 1
            token, prob = probs[i]
            acum += prob

        return token
