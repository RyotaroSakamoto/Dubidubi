import streamlit as st
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
plt.rcParams['font.family'] = 'Meiryo'
import requests
import json


st.title('猫ミームの歌詞自動生成')

st.write("マルコフ連鎖を用いて猫ミームの曲っぽい歌詞を作ってみよう")
st.markdown("---")
st.subheader("背景")
"""
昨今話題の猫ミームに出てくる曲Dubidubiduを聞いていたら、これマルコフ過程じゃね...?  
と思ったのでモデル化してみた。
"""
img = Image.open("image/MarcovCatmeme.png")
st.image(img,width=500)
"""
書き起こしたらそれっぽくなったので、この確率モデルを使ってDubidubiduっぽい歌詞生成するプログラムを作ってみました。  
このマルコフ連鎖は既約性(どの状態からスタートしても有限時間内に任意の状態に遷移可能)を持っているため任意の数で無限に歌詞を生成することが出来ます。
"""


st.markdown("---")
st.subheader("方法")
"""
マルコフ連鎖を使用して、n個の単語の歌詞を生成する。  
遷移確率行列を定義:
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


st.write("遷移モデルを可視化")
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
st.write("このモデルを用いて生成を行います")
st.markdown("---")

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

st.subheader("生成")


st.write("初期位置と歌詞の長さを選択して、生成ボタンを押してください")

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

n = st.slider("歌詞の長さを選択してください",0,100)
lis = generate_dubidubi(NEKOMEME_TRANS_PROB,labels=labels_jp,n=n,initial_state=w)
song = "".join(lis)
df = pd.DataFrame(lis, columns=['Word'])

# Count the occurrences of each word
word_counts = df['Word'].value_counts().reset_index()
word_counts.columns = ['Word', 'Count']
replacement_map = {
    "チピ": "chipi",
    "チャパ": "chapa",
    "ドゥビ": "dubi",
    "ダバ": "daba",
    "マヒ": "Mági",
    "コミ": "comi",
    "ブン": "boom"
}
# Replace the words in the DataFrame
word_counts = word_counts.replace(replacement_map)


def draw_plot(n,w,NEKOMEME_TRANS_PROB,labels_jp):
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


def draw_count_bar(words):
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Word', y='Count', data=words, palette='viridis')
    # plt.title('出現頻度')
    plt.xlabel('Word')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    sns.set(font='Meiryo')
    return st.pyplot(plt)



# 生成
if st.button('生成開始'):
    # st.text(song)
    for word in lis:
        st.write(word)
else:
    st.write('ボタンをクリックして生成してみよう')



st.markdown("---")
st.subheader("分析")
st.write("生成結果のデータを可視化して分析してみよう")

#分析
draw_count_bar(word_counts)

url = "https://api.tts.quest/v3/voicevox/synthesis"

# リクエストパラメータ
params = {
    "text": f"{song}",
    "speaker": 3
}
st.write(song)
response = requests.post(url, params=params)
response = response.content.decode() #バイト文字列からデコード
res_dic = json.loads(response)
mp3_url =  res_dic["mp3StreamingUrl"]
audio_data = requests.get(mp3_url).content
st.audio(audio_data)

st.markdown("---")
st.write("連絡等ありましたら下記メールアドレスまでお願いします")
st.write("mail:frandle256@gmail.com")
st.markdown("---")
st.subheader("参考にさせていただいたサイト様")
st.markdown("[Pythonで書くマルコフ連鎖の遷移確率](https://qiita.com/TeRa_YUKI/items/4edd10a06d1c606aaef4)")
st.markdown("[ウィキベディア:マルコフ連鎖](https://ja.wikipedia.org/wiki/%E3%83%9E%E3%83%AB%E3%82%B3%E3%83%95%E9%80%A3%E9%8E%96)")
st.markdown("[【道を開けろ】AIでコムドットやまと風の名言を作ろう（敬称略）](https://www.youtube.com/watch?v=x5AwzoQgt3E&t=109s&pp=ygUV44Oe44Or44Kz44OV6YCj6Y6W44CA)")


