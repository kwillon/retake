import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Функция загрузки данных
@st.cache_data
def load_data(csv_path):
    return pd.read_csv(csv_path)

# Пути к данным
df = load_data(r"C:\Users\Ксения\Project\retake\DashBords\M_3\Megy_cryst.csv")

st.title("Качество данных кристаллических структур")

# 1. Полнота данных
st.header("Полнота данных")
missing_values = df[['core_b', 'core_alpha', 'core_beta', 'core_gamma',
                     'shell_b', 'shell_alpha', 'shell_beta', 'shell_gamma']].isnull().mean() * 100
st.bar_chart(missing_values)

# Проверка случаев, когда orig_c2 = 0, но заполнены shell-данные
invalid_shell_data = df[(df['orig_c2'] == 0) & df[['shell_b', 'shell_alpha', 'shell_beta', 'shell_gamma']].notnull().any(axis=1)]
st.write(f"Количество некорректных данных оболочки: {invalid_shell_data.shape[0]}")

# 2. Анализ корректности значений
st.header("Корректность кристаллических параметров")
valid_angles = df[['core_alpha', 'core_beta', 'core_gamma', 'shell_alpha', 'shell_beta', 'shell_gamma']].apply(lambda col: col.between(0, 180)).all().all()
st.write("Все углы в допустимом диапазоне (0-180 градусов):", valid_angles)

# 3. Метрики API-запросов
st.header("Метрики API-запросов")
valid_structures = df[['space_group_core', 'space_group_shell']].notnull().mean() * 100
st.bar_chart(valid_structures)

# 4. Анализ выбросов
st.header("Выбросы в параметрах решетки")
def find_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return series[(series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))].count()

outliers = df[['core_b', 'shell_b']].apply(find_outliers)
st.bar_chart(outliers)

# 5. Визуализация распределения параметров
st.header("Распределение параметров решетки")
fig, ax = plt.subplots(figsize=(10, 6))

# Рисуем каждую гистограмму отдельно
for column, color in zip(['core_b', 'shell_b'], ['blue', 'green']):
    ax.hist(df[column].dropna(), bins=30, color=color, alpha=0.5, label=column, edgecolor='black')

ax.set_xlabel("Значения параметров")
ax.set_ylabel("Частота")
ax.legend()
st.pyplot(fig)

st.write("Анализ завершен")