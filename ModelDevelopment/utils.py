## This is For Freking Functions
#%%
import re as regex

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

modified_has_tag("#jokowi",)
#%%
def preprocessing_data(text):
	text = text.lower()
	