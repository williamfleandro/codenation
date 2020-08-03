#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import IPython


# In[2]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

from IPython.core.pylabtools import figsize


#figsize(12, 8)

#sns.set()


# In[3]:


countries = pd.read_csv("countries.csv")


# In[4]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.tail(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[5]:


# Sua análise começa aqui.
countries.info()


# # Convertendo em floats

# In[6]:


columns_convert = countries.columns.drop(["Country", "Region", "Population", "Area", "GDP"])
countries[columns_convert].head()


# In[7]:


countries[columns_convert] = countries[columns_convert].apply(lambda x:x.str.replace(",", ".").astype(float))


# In[8]:


countries.dtypes


# In[9]:


countries.head()


# In[48]:


countries["Country"] = countries["Country"].str.strip()
countries["Region"] = countries["Region"].str.strip()


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[55]:


def q1():
    # Retorne aqui o resultado da questão 1.    
    return sorted(countries["Region"].unique())


# In[56]:


q1()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[12]:


def q2():
    discretizer = KBinsDiscretizer(10, encode="ordinal").fit_transform(countries[["Pop_density"]])
    quantile = np.quantile(discretizer, 0.9)
    return int((discretizer > quantile).sum())


# In[13]:


q2()


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[14]:


countries[["Region", "Climate"]].isna().sum()


# In[15]:


countries.Climate.value_counts()


# In[17]:


countries.Climate.plot.box()


# In[18]:


countries.Climate.hist()


# In[19]:


countries.Climate.describe()


# In[61]:


def q3():
    # Retorne aqui o resultado da questão 3.
    climate = countries[["Climate"]].fillna(countries["Climate"].mean())
    one_hot_encoder = OneHotEncoder(sparse=False, dtype=np.int, handle_unknown="ignore")
    one_hot_region  = one_hot_encoder.fit_transform(countries[["Region"]])
    one_hot_climate = one_hot_encoder.fit_transform(climate)
    return int(one_hot_region.shape[1] + one_hot_climate.shape[1])


# In[62]:


q3()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[63]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[64]:


numercial_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("standard_scaler", StandardScaler())
])

columns = countries.columns[(countries.dtypes == float) | (countries.dtypes == int)]
numercial_pipeline.fit(countries[columns.values])


# In[65]:


def q4():
    # Retorne aqui o resultado da questão 4.
    numercial_pipeline.fit(countries[columns.values])
    pipeline_test_country = numercial_pipeline.transform([test_country[2:]])
    return float(round(pipeline_test_country[0][9], 3))


# In[66]:


q4()


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[ ]:


sns.boxplot(countries["Net_migration"].dropna())


# In[ ]:


sns.distplot(countries["Net_migration"].dropna())


# In[ ]:


countries["Net_migration"].isna().sum()


# In[ ]:


net_migration = countries["Net_migration"]
net_migration[:10]


# In[ ]:


def q5():
    # Retorne aqui o resultado da questão 4.
    q3, q1 = net_migration.quantile(0.75), net_migration.quantile(0.25) 
    IQR =  q3 - q1
    top_outliers = countries.query(f"Net_migration > {q3 + 1.5*IQR}").shape[0]
    lower_outliers = countries.query(f"Net_migration < {q1 - 1.5*IQR}").shape[0]    
    return (lower_outliers, top_outliers, False)


# In[ ]:


q5()


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[ ]:


categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[ ]:


def q6():
    # Retorne aqui o resultado da questão 4.
    count_vectorizer = CountVectorizer()
    newsgroup_counts = count_vectorizer.fit_transform(newsgroup.data)
    idx = count_vectorizer.vocabulary_.get("phone".lower())
    return int(newsgroup_counts[:, idx].toarray().sum())


# In[ ]:


q6()


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[ ]:


def q7():
    # Retorne aqui o resultado da questão 4.
    tfidf_vectorizer =  TfidfVectorizer()
    tfidf_vectorizer.fit(newsgroup.data)
    newsgroup_tfidf_vectorized = tfidf_vectorizer.transform(newsgroup.data)
    idx_phone = tfidf_vectorizer.vocabulary_.get("phone".lower())
    value = newsgroup_tfidf_vectorized[:, idx_phone].toarray().sum()
    return round(float(value), 3)


# In[ ]:


q7()

