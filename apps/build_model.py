import streamlit as st
# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import warnings
from gensim import corpora, models, similarities
import jieba
import re

def app(state):

    st.markdown('###Load Product_clean.csv')
    products = pd.read_csv("Product_clean.csv")

    st.markdown('#### View head 2')
    st.dataframe(products.head(2))

    #Store Session State
    st.session_state['products'] = products
    # In[4]:

    products.columns

    # In[5]:

    # Tokenize (split) the sentiment into words
    product_information_token = [[text for text in x.split()] for x in products.product_infomation]

    # In[6]:

    # Obtain the number of features based on dictionary: use corpora.Dictionary
    dictionary = corpora.Dictionary(product_information_token)

    # Store Session State
    st.session_state['dictionary'] = dictionary

    # In[7]:

    # List of features in dictionary
    # dictionary.token2id

    # In[8]:

    # Numbers of features (word) in dictionary
    st.markdown("### Numbers of features (word) in dictionary")
    feature_cnt = len(dictionary.token2id)
    st.markdown(f"$feature\_cnt={feature_cnt}$")

    # In[9]:

    # Obtain corpus based on dictionary (dense matrix: ma tran thua)
    st.markdown("### Obtain corpus based on dictionary (dense matrix: ma tran thua)")
    corpus = [dictionary.doc2bow(text) for text in product_information_token]
    st.write(corpus[0])

    # In[10]:

    # Use TF-IDF Model to process corpus, obtaining index
    st.markdown("### Use TF-IDF Model to process corpus, obtaining index")
    tfidf = models.TfidfModel(corpus)
    tfidf

    # Store Session State
    st.session_state['tfidf'] = tfidf
    # In[11]:

    # Tính toán sự tương tự trong ma trận thưa thớt
    index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=feature_cnt)

    # Store Session State
    st.session_state['index'] = index
    # In[12]:


