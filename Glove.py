from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

input_file = 'glove.txt.word2vec'

model = KeyedVectors.load_word2vec_format(input_file, binary=False)
result = model.most_similar(positive=["man", "woman"], negative=["man"], topn=1)
print(result)
