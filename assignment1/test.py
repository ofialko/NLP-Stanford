import codecs
import _pickle as pickle
from q3_sgd import load_saved_params
path = "utils/datasets/stanfordSentimentTreebank"
from utils.treebank import StanfordSentiment


dataset = StanfordSentiment()
sentences = dataset.sentences()
sentence = [codecs.decode(word, 'latin1') for word in sentences[0]]
" ".join(sentence)


dictionary = dict()
phrases = 0
with open(path + "/dictionary.txt", "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        splitted = line.split("|")
        dictionary[splitted[0].lower()] = int(splitted[1])
        phrases += 1


# sentences = []

# with open(path + "/datasetSentences.txt", "r", encoding='utf-8') as f:
#     for line in f:
#
#         splitted = line.strip().split()[1:]
#         # print(splitted)
#         # Deal with some peculiar encoding issues with this file
#
#         sentences += [[w.lower()
#                        for w in splitted]]
#         # sentences += [[codecs.decode(w.lower(), 'utf-8').encode('latin1')
#         #                for w in splitted]]
#
#
# sentences[1]
#
# with open('saved_params_25000.npy', 'rb') as f:
#     obj = pickle.load(f)
