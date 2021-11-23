#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import shutil
from typing import List
import re as regex

PATH_INDOLEM : str = ".\RawDataset\sentiment_indolem\data"

TARGET_INDOLEM : str = ".\PreprocessedDataset\indolem"
#%%
DataFrame = pd.DataFrame

#%%
def regex_dev(word : str) -> str:
    try:
        return regex.findall(r"dev",word)[0]
    except IndexError as e: 
        return None
    
def regex_train(word : str) -> str:
	try:    
		return regex.findall(r"train",word)[0]
	except IndexError as e : 
		return None
def regex_test(word:str) -> str:
    try:    
        return regex.findall(r"test",word)[0]
    except IndexError as e:
        return None
#%%
def get_train_dataset() -> List[str]: 
    list_train : List[str] = []
    for i in os.listdir(PATH_INDOLEM):
        if (regex_train(i) == "train"):
            list_train.append(i)
    return list_train

def get_validation_dataset() -> List[str]:
	list_val : List[str] = []
	for i in os.listdir(PATH_INDOLEM):
		if (regex_dev(i) == "dev"):
			list_val.append(i)
	return list_val

def get_test_dataset():
	list_test : List[str]  = []
	for i in os.listdir(PATH_INDOLEM):
		if (regex_test(i) == "test"):
			list_test.append(i)
	return list_test
#%%
def joined_train() -> DataFrame:
    train : List[str] = get_train_dataset()
    fullDataFrame : DataFrame = pd.DataFrame()
    for i in train:
        fullDataFrame = pd.concat([fullDataFrame,pd.read_csv(PATH_INDOLEM+"\\"+i)],axis=0)     
    return fullDataFrame

#%%
def joined_validation() -> DataFrame:
    validation : List[str] = get_validation_dataset()
    fullDataFrame : DataFrame = pd.DataFrame()
    for i in validation:
        fullDataFrame = pd.concat([fullDataFrame,pd.read_csv(PATH_INDOLEM+"\\"+i)],axis=0)
    return fullDataFrame

def joined_test() -> DataFrame:
    test : List[str] = get_test_dataset()
    fullDataFrame : DataFrame = pd.DataFrame()
    for i in test:
        fullDataFrame = pd.concat([fullDataFrame,pd.read_csv(PATH_INDOLEM+"\\"+i)],axis=0)
    return fullDataFrame 

def save_data(train_data : DataFrame, validation_data : DataFrame, test_data : DataFrame):
    train_data.to_csv(TARGET_INDOLEM+"\\train.csv",index=False)
    validation_data.to_csv(TARGET_INDOLEM+"\\validation.csv",index=False)
    test_data.to_csv(TARGET_INDOLEM+"\\test.csv",index=False)
    print("Data Saved Succesfully")
#%%
if __name__ == "__main__":
	train_data : DataFrame = joined_train()
	validation_data : DataFrame = joined_validation()
	test_data : DataFrame = joined_test()
	save_data(train_data,validation_data,test_data)
