#%%
from utils import preprocessing_data
from utils_DL import *
import tensorflow as tf
import pandas as pd
import numpy as np
import os

#* just for naming types convetion
array = np.array
#%%
if __name__ =="__main__":
    train_data = pd.read_csv("./PreprocessedDataset/indolem/train.csv")
    test_data = pd.read_csv("PreprocessedDataset/indolem/test.csv")
    validation_data = pd.read_csv("PreprocessedDataset/indolem/validation.csv")
    
    train_data_cleaned = train_data["sentence"].apply(lambda x:preprocessing_data(x))  
    validation_data_cleaned = validation_data["sentence"].apply(lambda x:preprocessing_data(x))
    
    X : array = train_data_cleaned.values
    Y : array = train_data["sentiment"].values
    max_features :int= 10000
    max_length :int= 50
    
    #* Params For WordEmbeddings
    min_count : int = 3
    vector_size : int = 50
    window : int = 5	
    sg : int = 1
    seed : int = 0
    
    #* Preparing the data for the model
    
    tokenizer,train_data_model,train_label_model = prepare_data(X, Y, max_features, max_length)
    print(train_data_model.shape)
    
    #* Preparing WordEmbeddings For Models
    if not os.path.exists("./utils.info/*.model"):
    	w2v_model : object = prepare_word_embeddings(X, min_count, vector_size, window, sg, seed)
    else:
        path : str = os.listdir("./utils.info/")[0]
        w2v_model : object = load_word_embeddings_model(path)
    
    embedding_matrix : array = createEmbeddingMatrix(tokenizer,w2v_model, max_length)
    print(embedding_matrix.shape)
    
    embedding_layer = prepare_embedding_layer(tokenizer, max_length, embedding_matrix)
    
    print(embedding_layer)
# %%
