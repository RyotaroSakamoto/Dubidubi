import streamlit as st
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Meiryo'

st.subheader('確率分布の実験')

'''
### 正規分布

母数（パラメータ）を変化させたときのグラフの変化の確認
'''

## 期待値と分散を指定
mu = st.sidebar.slider('正規分布の期待値', min_value=-5.0, max_value=5.0, step=0.01)
sigma = st.sidebar.slider('正規分布の分散', min_value=0, max_value=20, step=1)

## 標準正規分布の描画
x_1 = np.linspace(-10, 10, 100)
z = sp.stats.norm.pdf(x_1, mu, sigma)
fig_norm, ax1 = plt.subplots()
ax1.plot(x_1, z, label='std_norm')
ax1.legend()
st.pyplot(fig_norm)