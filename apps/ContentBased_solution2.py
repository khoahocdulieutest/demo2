#!/usr/bin/env python
# coding: utf-8
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



# st.set_page_config(
#     page_title="ContentBased solution2",
#     page_icon="👋",
# )
#
# st.write("# ContentBased solution2")
#
# st.sidebar.success("Select a demo above.")


# In[3]:


products = pd.read_csv("Product_clean.csv")
products.head(2)


# In[4]:


products.columns


# In[5]:


# Tokenize (split) the sentiment into words
product_information_token = [[text for text in x.split()] for x in products.product_infomation]


# In[6]:


# Obtain the number of features based on dictionary: use corpora.Dictionary
dictionary=corpora.Dictionary(product_information_token)


# In[7]:


# List of features in dictionary
#dictionary.token2id


# In[8]:


# Numbers of features (word) in dictionary
st.markdown("### Numbers of features (word) in dictionary")
feature_cnt=len(dictionary.token2id)
st.markdown(f"$feature_cnt={feature_cnt}$")


# In[9]:


# Obtain corpus based on dictionary (dense matrix: ma tran thua)
st.markdown("### Obtain corpus based on dictionary (dense matrix: ma tran thua)")
corpus=[dictionary.doc2bow(text) for text in product_information_token]
st.write(corpus[0])


# In[10]:


# Use TF-IDF Model to process corpus, obtaining index
st.markdown("### Use TF-IDF Model to process corpus, obtaining index")
tfidf = models.TfidfModel(corpus)
tfidf


# In[11]:


# Tính toán sự tương tự trong ma trận thưa thớt
index = similarities.SparseMatrixSimilarity(tfidf[corpus],num_features = feature_cnt)


# In[12]:


# When user choose one product: 1059892
st.markdown("### When user choose one product")
option = st.selectbox(
     'choose one product',
     products.item_id)

st.write('You selected:', option)
product_ID = 1059892
product_selection = products[products.item_id == option]
product_selection


# In[13]:


# sản phẩm đang xem
name_description_pre = product_selection['product_infomation'].to_string(index=False)
name_description_pre


# In[14]:


view_product = name_description_pre.lower().split()


# In[15]:


# Suggest other products for customers
def recommendation (view_product, dictionary, tfidf, index):
    # Convert search words into Sparse Vectors
    view_product = view_product.lower().split()
    kw_vector = dictionary.doc2bow(view_product)
    print("View product 's vector:")
    print(kw_vector)
    # Similarity calculation
    sim = index[tfidf[kw_vector]]
    
    # print result
    list_id = []
    list_score = []
    for i in range(len(sim)):
        list_id.append(i)
        list_score.append(sim[i])
    
    df_result = pd.DataFrame({'id': list_id,
                              'score': list_score})
    
    # 10 highest scores
    five_highest_score = df_result.sort_values(by='score', ascending=False).head(11)
    print("Five highest scores:")
    print(five_highest_score)
    print("Ids to list:")
    idToList = list(five_highest_score['id'])
    print(idToList)
    
    products_find = products[products.index.isin(idToList)]
    results = products_find[['item_id','name']]
    results = pd.concat([results, five_highest_score], axis=1).sort_values(by='score', ascending=False)
    return results


# In[16]:


results = recommendation(name_description_pre, dictionary, tfidf, index)


# In[17]:


# Recommender 5 similarities products for the selected product
results = results[results.item_id!=product_ID]
results


# In[18]:


# Save Content_Based_Filtering_Gensim_Dictionary


# In[19]:


dictionary.save("Content_Based_Filtering_Gensim_Dictionary.sav")


# In[20]:


dictionary.load("Content_Based_Filtering_Gensim_Dictionary.sav")


# In[21]:


#Solution 1 va Solution 2 co 2 danh sach goi y khac nhau


# In[ ]:




