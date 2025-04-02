import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
import textstat

# Загрузка данных
@st.cache_data
def load_data():
    return pd.read_csv("DashBords\M_4\megy_all1.csv")

df = load_data()

st.title("Data Quality Dashboard данных синтез-магнитные данные")

# Полнота данных
st.header("Полнота данных")
numeric_cols = ['sat_em_g', 'mr (emu/g)', 'coer_oe']
missing_values = df[numeric_cols].isnull().mean() * 100
st.bar_chart(missing_values)

# Анализ выбросов (IQR)
st.header("Выбросы в числовых данных")
def find_outliers(df, cols):
    outliers = {}
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
    return outliers
outliers = find_outliers(df, numeric_cols)
st.bar_chart(pd.Series(outliers))

# Анализ текстовых данных
st.header("Анализ текстовых данных")
text_data = df['Synthesis'].dropna()
unique_texts = text_data.nunique()

duplicate_synthesis = df.duplicated(subset=['Synthesis']).sum()
st.write(f"**Количество дубликатов в текстовых описаниях:** {duplicate_synthesis}")
st.write(f"**Количество уникальных описаний:** {unique_texts}")

# TF-IDF анализ
vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
tfidf_matrix = vectorizer.fit_transform(text_data)
st.write(f"**Размерность TF-IDF матрицы:** {tfidf_matrix.shape}")

# LDA анализ
lda = LatentDirichletAllocation(n_components=5, random_state=42)
topics = lda.fit_transform(tfidf_matrix)
st.write("**LDA тематическое моделирование выполнено**")

# Метрики текста
avg_text_length = text_data.apply(lambda x: len(x.split())).mean()
st.write(f"**Средняя длина описаний:** {avg_text_length:.2f} слов")

word_counter = Counter(" ".join(text_data).split())
common_phrases = sum([count for word, count in word_counter.items() if count > 5]) / len(word_counter)
st.write(f"**Доля повторяющихся фраз:** {common_phrases:.2f}")

readability_scores = text_data.apply(lambda x: textstat.flesch_reading_ease(x))
avg_readability = readability_scores.mean()
st.write(f"**Средний уровень читаемости текста (Flesch Reading Ease):** {avg_readability:.2f}")

st.success("Анализ данных завершен")