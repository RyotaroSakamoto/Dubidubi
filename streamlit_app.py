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
昨今話題の猫ミームに出てくる曲Dubidubiduを聞いていたら、これマルコフ過程じゃね...?
"""
img = Image.open("image/MarcovCatmeme.png")
st.image(img)
"""
書き起こしたらそれっぽくなったので、この確率モデルを使ってDubidubidoっぽい歌詞生成するプログラムを作ってみました。
"""
st.markdown("---")
st.subheader("方法")
"""
マルコフ連鎖を使用して、遷移確率行列からn単語までの歌詞を生成する。
"""



code = """

NEKOMEME_TRANS_PROB = np.array([
    [1/2,1/2,0,0,0,0,0], #チピ
    [0,1/2,1/2,0,0,0,0], #チャパ
    [0,0,1/2,1/4,0,0,1/4], #ドゥビ
    [0,0,0,1/2,1/2,0,0], #ダバ
    [0,0,0,0,0,1,0], #マヒ
    [0,0,1,0,0,0,0], #コミ
    [0.25,0,0,0,0,0,0.75], #ブン
])
NEKOMEME_TRANS_PROB

"""
st.code(code, language='python')

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

code = """

import graphviz
from graphviz import Digraph
import numpy as np

def Graphviz(prob_matrix, node_label):
    states = len(node_label)
    g = Digraph()

    for i in range(states):
        g.node(str(i), label=node_label[i])

    for i in range(states):
        for j in range(states):
            if prob_matrix[i, j] > 0:
                g.edge(str(i), str(j), label=str(round(prob_matrix[i, j], 2)))

    g.attr('node', fontname = 'Meiryo UI')
    g.attr('edge', fontname = 'Meiryo UI')
    return g

g = Graphviz(NEKOMEME_TRANS_PROB, labels)

g.view()
g.format = "png"
g.render("data/nekomeme.png",view=True)

"""
st.code(code, language='python')


img = Image.open("image/nekomeme.png")
st.image(img)


st.write("初期位置")
init_choice = st.selectbox("初期位置を選択してください",labels_jp)
match init_choice:
    case  "チピ":
        w = np.array([1,0,0,0,0,0,0])

n = 50
w = np.array([1,0,0,0,0,0,0])
lis = generate_dubidubi(NEKOMEME_TRANS_PROB,labels=labels_jp,n=n,initial_state=w)
for i in lis:
    st.write(i)