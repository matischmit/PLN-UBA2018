import math
import copy
import random
from languagemodeling.ngram import NGram
from languagemodeling.ngramaddone import AddOneNGram

class InterpolatedNGram(NGram):

    def __init__(self, n, sents, gamma=None, addone=False):

        self._n = n
        self._gamma = gamma
        development_data = []
        if gamma is None:
            len_held_out = int(0.1 * len(sents))
            development_data = sents[-len_held_out+1]
            sents = sents[0:len_held_out + 1]

        models = []
        for i in range(1, n + 1):
            if addone:
                models.append(AddOneNGram(i, copy.deepcopy(sents)))
            else:
                models.append(NGram(i, copy.deepcopy(sents)))

        self._ngram_models = models

        if gamma is None:
            self._gamma = self.gammaFromHeldOut(development_data)

        super().__init__(n, sents)

    def count(self, tokens):
        ngramsize = len(tokens)
        if ngramsize == 0:
            ngramsize = 1
        return self._ngram_models[ngramsize-1].count(tokens)

    def gammaFromHeldOut(self, data):

        #TODO
        return 1

    def cond_prob(self, token, prev_tokens=None):
        cond_ml = []        # maximum likelihood estimators
        lambdas = []        # calculo de lambdas
        for i in range(0,self._n - 1):
            if prev_tokens is None:
                prev_tokens_per_ngram = tuple()
            else:
                prev_tokens_per_ngram = prev_tokens

            c_prob = self._ngram_models[self._n-i-1].cond_prob(token, prev_tokens_per_ngram)  #para cada modelo de ngramas calculo la condicional
            if c_prob == -math.inf:
                c_prob = 0
            cond_ml.append(c_prob)


            current_lambda = (1 - sum(lambdas)) * self._ngram_models[self._n -(i+1)].count(tuple(prev_tokens[i:])) / (self._ngram_models[self._n -(i+1)].count(tuple(prev_tokens[i:])) + self._gamma)
            lambdas.append(current_lambda)
        cond_ml.append(self._ngram_models[0].cond_prob(token))
        lambdas.append(1 - sum(lambdas))

        prob = 0
        for i in range(len(cond_ml)):
            prob += lambdas[i] * cond_ml[i]

        return prob
