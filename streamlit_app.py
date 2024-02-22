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
st.subheader("この遷移確率を用いて生成を行う")

code = """
#n単語目までのDubidubido生成
def generate_dubidubi(transition_prob, labels, n, initial_state):
    # 単語リストを格納する配列を初期化
    word_list = []
    # 現在の状態を初期状態で設定
    current_state = np.random.choice(len(labels), p=initial_state)
    for _ in range(n):
        # 現在の状態に基づいて単語を選択し、リストに追加
        word_list.append(labels[current_state])
        # 遷移確率行列を使用して次の状態をランダムに選択
        current_state = np.random.choice(len(labels), p=transition_prob[current_state])
    return word_list
"""
#n単語目までのDubidubido生成
def generate_dubidubi(transition_prob, labels, n, initial_state):
    # 単語リストを格納する配列を初期化
    word_list = []
    # 現在の状態を初期状態で設定
    current_state = np.random.choice(len(labels), p=initial_state)
    for _ in range(n):
        # 現在の状態に基づいて単語を選択し、リストに追加
        word_list.append(labels[current_state])
        # 遷移確率行列を使用して次の状態をランダムに選択
        current_state = np.random.choice(len(labels), p=transition_prob[current_state])
    return word_list


init_choice = st.selectbox("初期位置を選択してください",labels_jp)
match init_choice:
    case  "チピ":
        w = np.array([1,0,0,0,0,0,0])
    case  "チャパ":
        w = np.array([0,1,0,0,0,0,0])
    case  "ドゥビ":
        w = np.array([0,0,1,0,0,0,0])
    case  "ダバ":
        w = np.array([0,0,0,1,0,0,0])
    case  "マヒ":
        w = np.array([0,0,0,0,1,0,0])
    case  "コミ":
        w = np.array([0,0,0,0,0,1,0])
    case  "ブン":
        w = np.array([0,0,0,0,0,0,1])

n = st.slider("単語の長さを選択してください",0,100)
lis = generate_dubidubi(NEKOMEME_TRANS_PROB,labels=labels_jp,n=n,initial_state=w)

# カウンタ更新関数
def update_counter():
    if 'counter' not in st.session_state:
        st.session_state.counter = 0  # セッション状態にカウンタを初期化
    st.session_state.counter += 1  # カウンタをインクリメント

# ボタンがクリックされたらカウンタを更新
if st.button('生成'):
    update_counter()

# 現在のカウンタ値を表示
if 'counter' in st.session_state:
    st.write(lis)
else:
    st.write('ボタンをクリックして生成')





#n回目までの単語の推移を計算
w_list = np.zeros((7, n))     #推移を記録する箱を作成
w_list[:,0] = w                 #初期値を記録
for k in range(1, n):
    w = w.dot(NEKOMEME_TRANS_PROB)        # 次期の確率の計算
    w_list[:,k] = w 


for i in range(7):
    plt.plot(w_list[i,:])
plt.grid()
plt.xlabel('回数')
plt.ylabel('確率')
plt.legend(labels_jp)
st.pyplot(plt)
