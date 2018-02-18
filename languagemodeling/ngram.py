# https://docs.python.org/3/library/collections.html
from collections import defaultdict
import math


class LanguageModel(object):

    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.

        sent -- the sentence as a list of tokens.
        """
        return 0.0

    def sent_log_prob(self, sent):
        """Log-probability of a sentence.

        sent -- the sentence as a list of tokens.
        """
        return -math.inf

    def log_prob(self, sents):
        result = 0.0
        for i, sent in enumerate(sents):
            lp = self.sent_log_prob(sent)
            if lp == -math.inf:
                return lp
            result += lp
        return result

    def cross_entropy(self, sents):
        log_prob = self.log_prob(sents)
        n = sum(len(sent) + 1 for sent in sents)  # count '</s>' events
        e = - log_prob / n
        return e

    def perplexity(self, sents):
        return math.pow(2.0, self.cross_entropy(sents))


class NGram(LanguageModel):

    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        assert n > 0
        self._n = n

        count = defaultdict(int)

        for sent in sents:
            self.stopAndStartSymbols(sent)

        for sent in sents:
            self.countFromNgram(count, n, sent)

        self._count = dict(count)

    def count(self, tokens):
        """Count for an n-gram or (n-1)-gram.

        tokens -- the n-gram or (n-1)-gram tuple.
        """
        return self._count.get(tokens, 0)

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """

        #P(t| prev_t) ~ #P(the|its water is so transparent that) = C(its water is so transparent that the) / C(its water is so transparent that)
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
            return self.count(tokens) / self.count(prev_tokens)

    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.

        sent -- the sentence as a list of tokens.
        """
        self.stopAndStartSymbols(sent)

        prob = 1
        for i in range(len(sent) - self._n + 1):
            ngram = sent[i:i+self._n]
            if self._n > 1:
                sent_prob = self.cond_prob(ngram[self._n - 1], ngram[0:self._n - 1])
                if sent_prob != -math.inf:
                    prob *= sent_prob
                else:
                    return 0
            else:
                prob *= self.cond_prob(ngram[self._n - 1])

        return prob

    def sent_log_prob(self, sent):
        """Log-probability of a sentence.

        sent -- the sentence as a list of tokens.
        """
        # WORK HERE!!
        self.stopAndStartSymbols(sent)

        prob = 0
        for i in range(len(sent) - self._n + 1):
            ngram = sent[i:i+self._n]
            sent_prob = self.cond_prob(ngram[self._n - 1], ngram[0:self._n - 1])
            if sent_prob != 0:
                prob += math.log(sent_prob, 2)
            else:
                return -math.inf
        return prob

    def countFromNgram(self, count, n, sent):
        for i in range(len(sent)):
            ngram, ngramminus = tuple(sent[i:i+n]), tuple(sent[i:i+n-1])
            count[ngram] += 1
            count[ngramminus] += 1

    def stopAndStartSymbols(self, sent):
        sent.append('</s>')
        for i in range(self._n - 1):
            sent.insert(0, '<s>')
