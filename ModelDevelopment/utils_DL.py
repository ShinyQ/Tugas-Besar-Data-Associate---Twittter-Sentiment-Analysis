#%%
import numpy as np
from typing import Dict,List,Tuple
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Embedding
from gensim.models import Word2Vec


array = np.array
tensor_array = tf.Tensor
def prepare_data(X : array , Y : array, max_features : int ,
                 max_length : int, oov_token : str="<OOV>",
                 padding_type : str = "post" )->Tuple[object,tensor_array,array ]:
    
    Y  = Y.astype(int)
    tokenizer = Tokenizer(num_words=max_features,oov_token=oov_token)
    tokenizer.fit_on_texts(X)
    x_sequences : tensor_array = tokenizer.texts_to_sequences(X)
    x_padded = pad_sequences(x_sequences, maxlen=max_length, padding=padding_type)
    Y = to_categorical(Y, num_classes=2)
    return tokenizer, x_padded, Y

def prepare_word_embeddings(raw_x : array, min_count : int, vector_size : int, window:int, sg:int,seed:int):
    raw_x : List[str]= [text.split() for text in raw_x]
    w2v_model : object = Word2Vec(raw_x, min_count=min_count, vector_size=vector_size, window=window, sg=sg, seed=seed)
    w2v_model.save(f"utils_info/gensim{vector_size}.model")
    return w2v_model
    

def createWordEmbeddings(w2v : object):
    return pd.DataFrame(w2v[w2v.wv.vocab], index=list(w2v.wv.vocab))

def createEmbeddingMatrix(tokenizer : object,w2v : object, max_features : int):
    embedding_matrix : array = np.zeros((len(tokenizer.word_index) + 1, int(max_features)))
    for word, i in tokenizer.word_index.items():
        try:
            embedding_vector = w2v.wv.get_vector(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        except:
            embedding_matrix[i] = np.array([0] * max_features)
    
    return embedding_matrix

def load_word_embeddings_model(path:str):
    return Word2Vec.load(path)

def prepare_embedding_layer(tokenizer : object, max_features:int, embedding_matrix : array, train : bool =False):
    embedding_layer = Embedding(input_dim=len(tokenizer.word_index) + 1,
    output_dim  = max_features,
    weights = [embedding_matrix],
    trainable  = train)
    
    return embedding_layer
# %%
