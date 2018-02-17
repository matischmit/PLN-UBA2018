import math
from languagemodeling.ngram import NGram

class AddOneNGram(NGram):

    def __init__(self, n, sents):
        super().__init__(n,sents)

        elements = []  #self.getSizeOfAlphabet(sents)
        for sent in sents:
            elementsInSent = [token for token in sent if token not in elements ]
            elements += elementsInSent
        self._size_of_alphabet = len(elements) + 1 - self._n

    def V(self):
        return self._size_of_alphabet


    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token with add one smooth.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        if prev_tokens is None:
            prev_tokens = ()
        else:
            if type(prev_tokens) is list:
                prev_tokens = tuple(prev_tokens)

        tokens = prev_tokens + (token,)

        if tokens is not tuple or prev_tokens is not tuple:
            tokens = tuple(tokens)
            prev_tokens = tuple(prev_tokens)


        if self.count(prev_tokens) is 0:
            return -math.inf
        else:
            return (self.count(tokens) + 1) / (self.count(prev_tokens) + self.V())