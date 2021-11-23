## This is For Freking Functions
#%%
import re as regex
import pandas as pd

BAKU = pd.read_csv("utils_info/kamus_alay.csv")
BAKU = BAKU.set_index("t_baku")["baku"].to_dict()
def replace_at_to_User(text: str) -> str:
    return regex.sub("@[a-zA-Z0-9]+","user",text)

def remove_tanda(text : str) -> str:
  text = regex.sub("[!\"#%$&\'@()*+,-./:;<=>?[\\]^_`{|}~]+","",text)
  return text

def remove_links(text : str) -> str:
    text = regex.sub("\S*:\S+", "",text)
    return text

def modified_has_tag(text : str)-> str:
    text =regex.sub("#", "kata kunci ", text).rstrip()
    return text

def stemming_data(text : str) -> str:
    return MPStemmer().stem_kalimat(text)

def map_to_baku(text, baku):
    text_copy = text.split()
    for i in range(len(text_copy)):
        if text_copy[i] in baku:
            text_copy[i] = baku[text_copy[i]]
    
    return " ".join(text_copy)

#%%
teks = "firman https://tololgaming.com/ ganteng @Jok @ren @sam #ganteng #kalem #tuman"
def preprocessing_data(text):
    text = text.lower();
    text = map_to_baku(text,BAKU)
    text = remove_links(text)
    text = replace_at_to_User(text)
    text = modified_has_tag(text)
    text = remove_tanda(text)
    #text = stemming_data(text)
    return text
preprocessing_data(teks)
# %%