import streamlit as st
import streamlit.components.v1 as components
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

# bootstrap 4 collapse example

def app(state):
    st.header('Recommendation Systems')

    products = st.session_state['products']
    dictionary = st.session_state['dictionary']
    tfidf = st.session_state['tfidf']
    index = st.session_state['index']

    # In[31]:

    def recommendation(view_product, dictionary, tfidf, index):
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
        results = products_find[['item_id', 'name']]
        results = pd.concat([results, five_highest_score], axis=1).sort_values(by='score', ascending=False)
        return results

    # Them vao
    products_temp = products.copy(deep=True)
    products_temp.set_index('item_id', inplace=True)
    products_dict = products_temp.to_dict('index')

    def format_func(item_id):
        return f'{item_id}-' + products_dict[item_id]['name']

    '''Xem chi tiet san pham'''
    def view_products(items):
        html = '''
            <style>
        input,
        textarea,
        button {
            height: 25px;
            margin: 0;
            padding: 10px;
            font-family: Raleway, sans-serif;
            font-weight: normal;
            font-size: 12pt;
            outline: none;
            border-radius: 0;
            background: none;
            border: 1px solid #282B33;
        }

        button,
        select {
            height: 45px;
            padding: 0 15px;
            cursor: pointer;
        }

        button {
            background: none;
            border: 1px solid black;
            margin: 25px 0;
        }

        button:hover {
            background-color: #282B33;
            color: white;
        }


        .tools {
            overflow: auto;
            zoom: 1;
        }

        .search-area {
            float: left;
            width: 60%;
        }

        .settings {
            display: none;
            float: right;
            width: 40%;
            text-align: right;
        }

        #view {
            display: none;
            width: auto;
            height: 47px;
        }

        #searchbutton {
            width: 60px;
            height: 47px;
        }

        input#search {
            width: 30%;
            width: calc(100% - 90px);
            padding: 10px;
            border: 1px solid #282B33;
        }

        @media screen and (max-width:400px) {
            .search-area {
                width: 100%;
            }
        }

        .products {
            width: 100%;
            font-family: Raleway;
        }

        .product {
            display: inline-block;
            width: calc(24% - 13px);
            margin: 10px 10px 30px 10px;
            vertical-align: top;
        }

        .product img {
            display: block;
            margin: 0 auto;
            width: auto;
            height: 200px;
            max-width: calc(100% - 20px);
            background-cover: fit;
            box-shadow: 0px 0px 7px 0px rgba(0, 0, 0, 0.8);
            border-radius: 2px;
        }

        .product-content {
            text-align: center;
        }

        .product h3 {
            font-size: 20px;
            font-weight: 600;
            margin: 10px 0 0 0;
        }

        .product h3 small {
            display: block;
            font-size: 16px;
            font-weight: 400;
            font-style: italic;
            margin: 7px 0 0 0;
        }

        .product .product-text {
            margin: 7px 0 0 0;
            color: #777;
        }

        .product .price {
            font-family: sans-serif;
            font-size: 16px;
            font-weight: 700;
        }

        .product .genre {
            font-size: 14px;
        }


        @media screen and (max-width:1150px) {
            .product {
                width: calc(33% - 23px);
            }
        }

        @media screen and (max-width:700px) {
            .product {
                width: calc(50% - 43px);
            }
        }

        @media screen and (max-width:400px) {
            .product {
                width: 100%;
            }
        }

        /* TABLE VIEW */

        @media screen and (min-width:401px) {
            .settings {
                display: block;
            }
            #view {
                display: inline;
            }
            .products-table .product {
                display: block;
                width: auto;
                margin: 10px 10px 30px 10px;
            }
            .products-table .product .product-img {
                display: inline-block;
                margin: 0;
                width: 120px;
                height: 120px;
                vertical-align: middle;
            }
            .products-table .product img {
                width: auto;
                height: 120px;
                max-width: 120px;
            }
            .products-table .product-content {
                text-align: left;
                display: inline-block;
                margin-left: 20px;
                vertical-align: middle;
                width: calc(100% - 145px);
            }
            .products-table .product h3 {
                margin: 0;
            }
        }
    </style>
            <div class="products products-table">
                           
            
            '''
        #st.write(items.iterrows())
        for index, row in items.iterrows():
            item = products_dict[row['item_id']]
            item_html = '''
            <div class="product">
            <div class="product-img">
                <a href="{0}" target="_blank"><img src="{0}"></a>
            </div>
            <div class="product-content">
                <h3>
                    <a href="{5}" target="_blank">{1}</a>
                    <small>{2}</small>
                </h3>
                <p class="product-text price">Price: {3}</p>
                <p class="product-text genre">Rating: {4}</p>
            </div>
            </div>
            '''.format(item['image'],item['name'],item['product_infomation'], item['price'], item['rating'], item['url'])
            html = html + item_html

        html = html + "</div>"
        st.components.v1.html(html,height=600, scrolling=True)

    # When user choose one product: 1059892
    st.markdown("### Please choose one product")
    item_option = st.selectbox(
        'Choose one product',
        options=list(products_dict.keys()),
        format_func=format_func)

    st.write('You selected:', f'{format_func(item_option)}')
    product_ID = item_option
    product_selection = products[products.item_id == item_option]
    product_selection

    # In[13]:

    # sản phẩm đang xem
    name_description_pre = product_selection['product_infomation'].to_string(index=False)
    name_description_pre

    # In[14]:

    view_product = name_description_pre.lower().split()

    # In[15]:

    results = recommendation(name_description_pre, dictionary, tfidf, index)

    # In[17]:

    # Recommender 5 similarities products for the selected product
    st.markdown('### Recommender 5 similarities products for the selected product')
    results = results[results.item_id != product_ID]
    st.dataframe(results)
    view_products(results)
    #st.write(detail)
