import streamlit as st
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams['font.family'] = 'Meiryo'

st.header('Dubidubido(猫ミーム)歌詞自動生成')
st.subheader("マルコフ連鎖を用いたDubidubidoっぽい歌詞生成")
print()
st.markdown("---")
st.subheader("背景")
"""
昨今話題の猫ミームに出てくる曲Dubidubiduを聞いていた際、ん？これマルコフ過程じゃね...?
と思って書き起こしたらそれっぽくなった(図1)ので、この確率モデルを使ってDubidubidoっぽい歌詞生成するプログラムを作ってみました。
"""
#遷移確率行列を作成
NEKOMEME_TRANS_PROB = np.array([
    [1/2,1/2,0,0,0,0,0], #チピ
    [0,1/2,1/2,0,0,0,0], #チャパ
    [0,0,1/2,1/4,0,0,1/4], #ドゥビ
    [0,0,0,1/2,1/2,0,0], #ダバ
    [0,0,0,0,0,1,0], #マヒ
    [0,0,1,0,0,0,0], #コミ
    [0.25,0,0,0,0,0,0.75], #ブン
])
labels = ["chipi", "chapa", "dubi", "daba", "Mági", "comi", "boom"]
labels_jp = ["チピ", "チャパ", "ドゥビ", "ダバ", "マヒ", "コミ", "ブン"]

NEKOMEME_TRANS_PROB

img = Image.open("image/nekomeme.png")
st.image(img)