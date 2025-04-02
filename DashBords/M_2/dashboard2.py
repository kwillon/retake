import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
@st.cache_data
def load_data(csv_path, json_folder):
    df = pd.read_csv(csv_path)
    
    # Загрузка JSON-файлов
    def load_json(file):
        path = os.path.join(json_folder, file)
        if os.path.exists(path) and file.endswith('.json'):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    df['json_c1'] = df['json_files_c1'].apply(lambda x: load_json(x) if pd.notna(x) else None)
    df['json_c2'] = df['json_files_c2'].apply(lambda x: load_json(x) if pd.notna(x) else None)
    
    return df

# Пути к данным
df = load_data(r"DashBords\M_2\Megy_json_files.csv.csv",
                r"DashBords\M_2\json")

st.title("Data Quality Dashboard для json файлов")

# Полнота данных
st.header("Полнота данных")
missing_values = df.isnull().mean() * 100
st.bar_chart(missing_values)

# Дубликаты
st.header("Дубликаты")
duplicates = df.duplicated(subset=['orig_c1', 'orig_c2', 'link']).sum()
st.write(f"Количество дубликатов: {duplicates}")

# Выбросы
st.header("Выбросы в числовых данных")
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

# Визуализация
st.header("Распределение числовых данных")
numeric_cols = df.select_dtypes(include=[np.number]).columns
fig, ax = plt.subplots(figsize=(10, 6))
df[numeric_cols].hist(ax=ax, bins=30, color='skyblue', edgecolor='black')
st.pyplot(fig)

# Анализ JSON
st.header("Анализ JSON данных")
df['json_c1_valid'] = df['json_c1'].apply(lambda x: x is not None and isinstance(x, dict))
df['json_c2_valid'] = df['json_c2'].apply(lambda x: x is not None and isinstance(x, dict))
st.write("Корректность JSON данных:")
st.write(df[['json_c1_valid', 'json_c2_valid']].mean() * 100)

# Проверка JSON-файлов
st.header("Анализ JSON данных")

df['json_c1_missing'] = df.apply(lambda row: pd.notna(row['orig_c1']) and row['orig_c1'] != '0' and row['json_c1'] is None, axis=1)
df['json_c2_missing'] = df.apply(lambda row: pd.notna(row['orig_c2']) and row['orig_c2'] != '0' and row['json_c2'] is None, axis=1)

missing_json_c1 = df['json_c1_missing'].sum()
missing_json_c2 = df['json_c2_missing'].sum()
st.write(f"Отсутствующие JSON для c1: {missing_json_c1}")
st.write(f"Отсутствующие JSON для c2: {missing_json_c2}")

# Проверка структуры JSON
def check_json_structure(json_data):
    if not json_data:
        return False
    if isinstance(json_data, dict) and len(json_data) > 0:
        return True
    return False

df['json_c1_valid'] = df['json_c1'].apply(check_json_structure)
df['json_c2_valid'] = df['json_c2'].apply(check_json_structure)

st.write("Корректность JSON данных:")
st.write(df[['json_c1_valid', 'json_c2_valid']].mean() * 100)

# Визуализация распределения JSON-данных
st.header("Распределение JSON файлов")
fig, ax = plt.subplots(figsize=(8, 5))
df[['json_c1_valid', 'json_c2_valid']].mean().plot(kind='bar', color=['blue', 'red'], ax=ax)
plt.xticks(rotation=0)
st.pyplot(fig)