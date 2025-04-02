import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
@st.cache_data
def load_data():
    return pd.read_csv(r"DashBords\M_1\Copy of Megy_data_durty- July 11, 2024, 5_57 PM - Sheet1.csv")

df = load_data()

# Удаление ненужных столбцов
df = df.drop(columns=['c1', 'c2', 'c3'])

st.title("Data Quality Dashboard")

# Полнота данных
st.header("Полнота данных")
completeness_data = df.isnull().mean() * 100
st.bar_chart(completeness_data)

# Дубликаты
duplicates = df.duplicated(subset=['orig_c1', 'orig_c2', 'link', 'curie_temperature_k', 
                                   'coer_oe', 'sat_em_g', 'mr (emu/g)',
                                   'crystal_structure', 'x', 'shape',
                                   'h_range_max_koe', 'crystal_structure']).sum()
st.write(f"Количество дубликатов: {duplicates}")

# Выбросы
def find_outliers(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
    return outliers

outliers = find_outliers(df)
st.bar_chart(pd.Series(outliers))

# Проверка диапазонов значений
def check_ranges(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    ranges = {}
    for col in numeric_cols:
        ranges[col] = {'min': df[col].min(), 'max': df[col].max()}
    return ranges

ranges = check_ranges(df)
st.write(pd.DataFrame.from_dict(ranges, orient='index'))

# Тепловая карта пропущенных значений
st.header("Тепловая карта пропущенных значений")
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', xticklabels=df.columns, yticklabels=False)
st.pyplot(fig)

# Гистограммы числовых данных
st.header("Распределение числовых данных")
fig, ax = plt.subplots(figsize=(15, 10))
df.select_dtypes(include=[np.number]).hist(ax=ax, bins=30, color='skyblue', edgecolor='black')
st.pyplot(fig)

# Boxplot для выбросов
st.header("Распределение выбросов в числовых данных")
fig, ax = plt.subplots(figsize=(12, 8))
sns.boxplot(data=df.select_dtypes(include=[np.number]), orient="h", palette="Set2")
st.pyplot(fig)

# Корреляционная матрица
st.header("Корреляция между числовыми переменными")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=False, cmap='coolwarm', cbar=True, linewidths=0.5)
st.pyplot(fig)

st.success("Анализ завершен")