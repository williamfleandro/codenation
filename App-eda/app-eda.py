import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import altair as alt

def main():
    st.title('Analise de dados exploratória')
    st.subheader('por: William Ferreira Leandro')

    st.image('https://media.giphy.com/media/QxZ0nbcVgMlPlnfZos/giphy.gif', width=250)
    separador = st.radio("Escolha sua base de dados: ", (',',';'))
    file = st.file_uploader('Localize seu arquivo (.csv)', type='csv')

    if file is not None:
        slider = st.slider('Conhecendo os dados: ', 1, 100)
        df = pd.read_csv(file, sep=separador)
        st.dataframe(df.head(slider))

        st.subheader('Analise Estatística Descritiva')

        linhas = st.checkbox('Números de Linhas: ')
        if linhas:
            st.markdown(df.shape[0])

        atributos = st.checkbox('Número de Colunas: ')
        if atributos:
            st.markdown(df.shape[1])

        aux = pd.DataFrame({"Colunas": df.columns, 'Tipos':df.dtypes})
        colunas_numericas = list(aux[aux['Tipos'] != 'object']['Colunas'])
        colunas_object = list(aux[aux['Tipos'] == 'object']['Colunas'])
        colunas = list(df.columns)
        col = st.selectbox('Selecione a coluna: ', colunas_numericas)

        if col is not None:
            st.markdown('Selecione o que deseja analisar: ')

            mean = st.checkbox('Média')
            if mean:
                st.markdown(df[col].mean())

            median = st.checkbox('Mediana')
            if median:
                st.markdown(df[col].median())

            moda = st.checkbox('Moda')
            if moda:
                moda = stats.mode(df[col])
                st.markdown(moda[0])

            desvio_pad = st.checkbox('Desvio Padrão')
            if desvio_pad:
                st.markdown(df[col].std())

            kurtosis = st.checkbox('Achatamento da curva')
            if kurtosis:
                st.markdown(df[col].kurtosis())

            skewness = st.checkbox('Assimetria da Distribuição')
            if skewness:
                st.markdown(df[col].skew())

            st.subheader('Data visualization')
            st.markdown('Selecione a visualização')

            graf_bar = st.checkbox('Gráfico de Barras')
            if graf_bar:
                graf_bar = pd.DataFrame(df.groupby(col))
                st.bar_chart(graf_bar)

            graf_linhas = st.checkbox('Gráfico de Linhas')
            if graf_linhas:
                graf_linhas= pd.DataFrame(df.groupby(col))
                st.line_chart(graf_linhas)

            graf_area = st.checkbox('Gráfico de Area')
            if graf_area:
                graf_area = pd.DataFrame(df.groupby(col))
                st.area_chart(graf_area)

            graf_hist = st.checkbox('Gráfico de Histograma')
            if graf_hist:
                x = df[col]
                fig2, ax2 = plt.subplots()
                ax2.set_title("Gráfico de Barras " +str(col))
                plt.hist(x, bins=20)
                st.pyplot()

            graf_box = st.checkbox('Gráfico de Boxplot')
            if graf_box:
                x = df[col]
                fig2, ax2 = plt.subplots()
                ax2.set_title("Gráfico de Boxplot " +str(col))
                plt.boxplot(x, notch=True)
                st.pyplot()

if __name__ == '__main__':
    main()





