import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="JRA競馬予想AI", page_icon="🏇")
st.title("🏇 マイJRA競馬予想AI（自動入力対応版）")

# モデルとデータベースの読み込み
@st.cache_resource
def load_all():
    model_v2 = joblib.load("jra_honkaku_model_v2.pkl")
    model_ana = joblib.load("jra_anauma_model.pkl")
    horse_db = pd.read_csv("horse_db.csv").set_index("馬名")
    jockey_db = pd.read_csv("jockey_db.csv").set_index("騎手")
    return model_v2, model_ana, horse_db, jockey_db

model_v2, model_ana, horse_db, jockey_db = load_all()

mode = st.radio(
    "使用するAIを選択してください",
    ["💎 穴馬発掘AI（実力のみ・AUC 0.735）", "🔥 総合・的中重視AI（オッズ込み・AUC 0.808）"]
)

st.divider()
st.subheader("1. 馬名・騎手名の選択（過去データを自動取得）")

col_search1, col_search2 = st.columns(2)
with col_search1:
    # 検索または手入力できるセレクトボックス
    horse_list = ["（新規・データ外の馬を手入力）"] + sorted(horse_db.index.tolist())
    selected_horse = st.selectbox("馬名を選択（文字入力で絞り込み可）", horse_list)
    if selected_horse == "（新規・データ外の馬を手入力）":
        horse_name = st.text_input("馬名を入力", "ラフターラインズ")
        h_data = None
    else:
        horse_name = selected_horse
        h_data = horse_db.loc[selected_horse]
        st.caption(f"✅ 過去データ取得済（前走{int(h_data['前走着順'])}着 / 上り{h_data['前走上り3F']}秒）")

with col_search2:
    jockey_list = ["（データ外の騎手・手動設定）"] + sorted(jockey_db.index.tolist())
    selected_jockey = st.selectbox("騎手を選択（文字入力で絞り込み可）", jockey_list)
    if selected_jockey == "（データ外の騎手・手動設定）":
        default_j_rate = 25
    else:
        default_j_rate = int(jockey_db.loc[selected_jockey]['騎手_直近複勝率'] * 100)
        st.caption(f"✅ 騎手データ取得済（直近複勝率 {default_j_rate}%）")

# 取得したデータを初期値としてセット（データがない場合は標準値）
def_sex = str(h_data['性別']) if (h_data is not None and str(h_data['性別']) in ["牡", "牝", "セ"]) else "牝"
def_age = int(min(max(h_data['前走時馬齢'], 2), 10)) if h_data is not None else 3
def_weight = int(min(max(h_data['前走馬体重'], 380), 600)) if h_data is not None else 470
def_prev_rank = int(min(max(h_data['前走着順'], 1), 18)) if h_data is not None else 3
def_prev_agari = float(min(max(h_data['前走上り3F'], 31.0), 45.0)) if h_data is not None else 34.0
def_prev_dist = int(h_data['前走距離']) if h_data is not None else 2000
def_avg_rank3 = float(min(max(h_data['過去3走平均着順'], 1.0), 18.0)) if h_data is not None else 3.5

st.divider()
st.subheader("2. 今回のレース条件・出馬表データ")

col1, col2 = st.columns(2)
with col1:
    place = st.selectbox("競馬場", ["東京", "中山", "京都", "阪神", "中京", "新潟", "福島", "小倉", "札幌", "函館"])
    surface = st.selectbox("芝・ダート", ["芝", "ダート"])
    condition = st.selectbox("馬場状態", ["良", "稍重", "重", "不良"])
    distance = st.number_input("今回の距離 (m)", min_value=1000, max_value=3600, value=2200, step=100)
    waku = st.number_input("枠番", min_value=1, max_value=8, value=7)
    umaban = st.number_input("馬番", min_value=1, max_value=18, value=15)

with col2:
    sex = st.selectbox("性別", ["牡", "牝", "セ"], index=["牡", "牝", "セ"].index(def_sex))
    age = st.number_input("馬齢", min_value=2, max_value=10, value=def_age)
    kinryo = st.number_input("斤量 (kg)", min_value=48.0, max_value=62.0, value=54.0, step=0.5)
    weight = st.number_input("馬体重 (kg)", min_value=380, max_value=600, value=def_weight, step=2)
    weight_diff = st.number_input("馬体重増減 (kg)", min_value=-30, max_value=30, value=0, step=2)
    tansho = st.number_input("単勝オッズ (倍)", min_value=1.0, max_value=999.9, value=12.6, step=0.1)
    fukusho = st.number_input("複勝オッズ目安 (倍) ※下限値", min_value=1.0, max_value=100.0, value=3.9, step=0.1)
    ninki = st.number_input("人気", min_value=1, max_value=18, value=7)

st.subheader("3. 過去成績・騎手データ（自動反映済・微調整可）")
col3, col4 = st.columns(2)
with col3:
    prev_rank = st.number_input("前走着順", min_value=1, max_value=18, value=def_prev_rank)
    prev_agari = st.number_input("前走上がり3F (秒)", min_value=31.0, max_value=45.0, value=def_prev_agari, step=0.1)
    # 今回の距離と前走距離の差を自動計算！
    auto_dist_diff = distance - def_prev_dist
    prev_dist_diff = st.number_input("前走との距離差 (m) ※自動計算済", min_value=-1600, max_value=1600, value=auto_dist_diff, step=100)
with col4:
    avg_rank3 = st.number_input("過去3走の平均着順", min_value=1.0, max_value=18.0, value=def_avg_rank3, step=0.5)
    jockey_rate = st.slider("騎手の直近複勝率 (%)", min_value=0, max_value=80, value=default_j_rate) / 100.0

if st.button("🚀 AIで確率＆回収期待値を判定する", type="primary", use_container_width=True):
    data = {
        '枠番': [waku],
        '馬番': [umaban],
        '馬齢': [age],
        '斤量': [kinryo],
        '距離(m)': [distance],
        '単勝': [tansho],
        '人気': [ninki],
        '馬体重': [weight],
        '場体重増減': [weight_diff],
        '前走着順': [prev_rank],
        '前走上り3F': [prev_agari],
        '前走距離差': [prev_dist_diff],
        '過去3走平均着順': [avg_rank3],
        '騎手_直近複勝率': [jockey_rate],
        '競馬場名': [place],
        '芝・ダート区分': [surface],
        '馬場状態1': [condition],
        '性別': [sex]
    }
    df_input = pd.DataFrame(data)
    for col in ['競馬場名', '芝・ダート区分', '馬場状態1', '性別']:
        df_input[col] = df_input[col].astype('category')

    if "総合" in mode:
        features = ['枠番', '馬番', '馬齢', '斤量', '距離(m)', '単勝', '人気', '馬体重', '場体重増減',
                    '前走着順', '前走上り3F', '前走距離差', '過去3走平均着順', '競馬場名', '芝・ダート区分', '馬場状態1', '性別']
        prob = model_v2.predict_proba(df_input[features])[:, 1][0] * 100
    else:
        features = ['枠番', '馬番', '馬齢', '斤量', '距離(m)', '馬体重', '場体重増減',
                    '前走着順', '前走上り3F', '前走距離差', '過去3走平均着順', '騎手_直近複勝率',
                    '競馬場名', '芝・ダート区分', '馬場状態1', '性別']
        prob = model_ana.predict_proba(df_input[features])[:, 1][0] * 100

    fukusho_ev = prob * fukusho
    tansho_ev = (prob * 0.35) * tansho

    st.divider()
    st.subheader(f"📊 {horse_name} のAI分析レポート")

    m1, m2, m3 = st.columns(3)
    m1.metric("AI予測 複勝率", f"{prob:.1f}%")
    m2.metric("複勝 期待値", f"{fukusho_ev:.0f}%", delta=f"{fukusho_ev - 100:.0f}% (基準100%)")
    m3.metric("単勝 推定期待値", f"{tansho_ev:.0f}%", delta=f"{tansho_ev - 100:.0f}% (基準100%)")

    if fukusho_ev >= 110 or tansho_ev >= 120:
        st.success("💰 **【超お宝・妙味大】期待値100%超え！積極的に狙うべき買い目です！**")
    elif fukusho_ev >= 100 or tansho_ev >= 100:
        st.info("✅ **【妙味あり】期待値100%ライン突破！プラス収支が見込める狙い目です。**")
    elif prob >= 45.0:
        st.warning("⚖️ **【実力上位だが人気過剰】** 3着以内率は高いですが、オッズが低いため軸向きです。")
    else:
        st.error("❄️ **【見送り推奨】** 確率・オッズともに妙味が薄いため、今回は見送りが無難です。")
