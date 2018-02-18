import math
import random
from languagemodeling.ngram import NGram

class InterpolatedNGram(NGram):

    def __init__(self, n, sents, gamma=None, addone=True):

        self._gamma = gamma
        if gamma is None:
            len_held_out = int(0.1 * len(sents))
            development_data = sents[-len_held_out]
            sents = sents[0:len_held_out+1]
            self.gamma = self.gammaFromHeldOut(development_data)


        models = []
        for i in range(1, n + 1):
            models.append(NGram(i, sents))

        self._ngram_models = models

        # if is_addone:
        #     models.append(AddOneNGram(1, sents))
        # else:
        #     models.append(NGram(1, sents))

        super().__init__(n, sents)



    def gammaFromHeldOut(self, data):
        #TODO
        return 1

    def count(self, tokens):
        return self._ngram_models[self._n-1].count(tokens)

    def cond_prob(self, token, prev_tokens=None):
        cond_ml = []        # maximum likelihood estimators
        lambdas = []        # calculo de lambdas
        for i in range(0,self._n - 1):
            if prev_tokens is None:
                prev_tokens_per_ngram = tuple()
            else:
                prev_tokens_per_ngram = prev_tokens

            cond_prob = self._ngram_models[self._n -i-1].cond_prob(token, prev_tokens_per_ngram)  #para cada modelo de ngramas calculo la condicional
            if cond_prob == -math.inf:
                cond_prob = 0
            cond_ml.append(cond_prob)

            current_lambda = (1 - sum(lambdas)) * self._ngram_models[self._n -(i+1)].count(prev_tokens[i:]) / (self._ngram_models[self._n -(i+1)].count(prev_tokens[i:]) + self._gamma)
            lambdas.append(current_lambda)

        cond_ml.append(self._ngram_models[0].cond_prob(token))
        lambdas.append(1 - sum(lambdas))

        prob = 0
        for i in range(len(cond_ml)):
            prob += lambdas[i] * cond_ml[i]

        return prob