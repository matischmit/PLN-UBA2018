"""Train an n-gram model.

Usage:
  train.py [-m <model>] -n <n> -o <file>
  train.py -h | --help

Options:
  -n <n>        Order of the model.
  -m <model>    Model to use [default: ngram]:
                  ngram: Unsmoothed n-grams.
                  addone: N-grams with add-one smoothing.
                  inter: N-grams with interpolation smoothing.
  -o <file>     Output model file.
  -h --help     Show this screen.
"""
pattern = r'''(?x)    # set flag to allow verbose regexps
   (?:\d{1,3}(?:\.\d{3})+)  # numbers with '.' in the middle
   | (?:[Ss]r\.|[Ss]ra\.|art\.)  # common spanish abbreviations
   | (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A.
   | \w+(?:-\w+)*        # words with optional internal hyphens
   | \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
   | \.\.\.            # ellipsis
   | [][.,;"'?():-_`]  # these are separate tokens; includes ], [
'''


from docopt import docopt
import pickle

from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(pattern)


from languagemodeling.ngram import NGram


# models = {
#     'ngram': NGram,
#     'addone': AddOneNGram,
#     'inter': InterpolatedNGram,
# }


if __name__ == '__main__':
    opts = docopt(__doc__)

    corpusReader = PlaintextCorpusReader(".", "southpark.txt", word_tokenizer=tokenizer)
    sents = corpusReader.sents()
    n = int(opts['-n'])
    model = NGram(n, sents)

    #model_class = models[opts['-m']]
    #model = model_class(n, sents)

    filename = opts['-o']
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()
