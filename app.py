import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="JRA競馬予想AI", page_icon="🏇")
st.title("🏇 マイJRA競馬予想AI")

@st.cache_resource
def load_models():
    model_v2 = joblib.load("jra_honkaku_model_v2.pkl")
    model_ana = joblib.load("jra_anauma_model.pkl")
    return model_v2, model_ana

model_v2, model_ana = load_models()

mode = st.radio(
    "使用するAIを選択してください",
    ["🔥 総合・的中重視AI（オッズ込み・AUC 0.808）", "💎 穴馬発掘AI（実力のみ・AUC 0.735）"]
)

st.divider()
st.subheader("1. 出走馬・オッズデータの入力")

col1, col2 = st.columns(2)
with col1:
    horse_name = st.text_input("馬名", "サンプルホース")
    place = st.selectbox("競馬場", ["東京", "中山", "京都", "阪神", "中京", "新潟", "福島", "小倉", "札幌", "函館"])
    surface = st.selectbox("芝・ダート", ["芝", "ダート"])
    condition = st.selectbox("馬場状態", ["良", "稍重", "重", "不良"])
    distance = st.number_input("今回の距離 (m)", min_value=1000, max_value=3600, value=1600, step=100)
    waku = st.number_input("枠番", min_value=1, max_value=8, value=1)
    umaban = st.number_input("馬番", min_value=1, max_value=18, value=1)

with col2:
    sex = st.selectbox("性別", ["牡", "牝", "セ"])
    age = st.number_input("馬齢", min_value=2, max_value=10, value=3)
    kinryo = st.number_input("斤量 (kg)", min_value=48.0, max_value=62.0, value=56.0, step=0.5)
    weight = st.number_input("馬体重 (kg)", min_value=380, max_value=600, value=480, step=2)
    weight_diff = st.number_input("馬体重増減 (kg)", min_value=-30, max_value=30, value=0, step=2)
    tansho = st.number_input("単勝オッズ (倍)", min_value=1.0, max_value=999.9, value=8.5, step=0.1)
    fukusho = st.number_input("複勝オッズ目安 (倍) ※下限値", min_value=1.0, max_value=100.0, value=2.5, step=0.1)
    ninki = st.number_input("人気", min_value=1, max_value=18, value=4)

st.subheader("2. 過去成績・騎手データ")
col3, col4 = st.columns(2)
with col3:
    prev_rank = st.number_input("前走着順", min_value=1, max_value=18, value=3)
    prev_agari = st.number_input("前走上がり3F (秒)", min_value=31.0, max_value=45.0, value=34.5, step=0.1)
    prev_dist_diff = st.number_input("前走との距離差 (m) ※延長はプラス", min_value=-1600, max_value=1600, value=0, step=100)
with col4:
    avg_rank3 = st.number_input("過去3走の平均着順", min_value=1.0, max_value=18.0, value=4.0, step=0.5)
    jockey_rate = st.slider("騎手の直近複勝率 (%)", min_value=0, max_value=80, value=25) / 100.0

# 3. 予測と期待値判定の実行
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
        prob = model_v2.predict_proba(df_input[features])[0, 1] * 100
    else:
        features = ['枠番', '馬番', '馬齢', '斤量', '距離(m)', '馬体重', '場体重増減',
                    '前走着順', '前走上り3F', '前走距離差', '過去3走平均着順', '騎手_直近複勝率',
                    '競馬場名', '芝・ダート区分', '馬場状態1', '性別']
        prob = model_ana.predict_proba(df_input[features])[0, 1] * 100

    # 期待値（回収率）の計算
    fukusho_ev = prob * fukusho  # 複勝期待値 (%)
    est_win_prob = prob * 0.35   # 複勝率から推定した1着勝率 (%)
    tansho_ev = est_win_prob * tansho  # 単勝期待値 (%)

    st.divider()
    st.subheader(f"📊 {horse_name} のAI分析レポート")

    # 3つの指標をカード形式で大きく表示
    m1, m2, m3 = st.columns(3)
    m1.metric("AI予測 複勝率", f"{prob:.1f}%")
    m2.metric("複勝 期待値", f"{fukusho_ev:.0f}%", delta=f"{fukusho_ev - 100:.0f}% (基準100%)")
    m3.metric("単勝 推定期待値", f"{tansho_ev:.0f}%", delta=f"{tansho_ev - 100:.0f}% (基準100%)")

    # 総合買い目判定メッセージ
    if fukusho_ev >= 110 or tansho_ev >= 120:
        st.success("💰 **【超お宝・妙味大】期待値100%超え！積極的に狙うべき買い目です！**")
    elif fukusho_ev >= 100 or tansho_ev >= 100:
        st.info("✅ **【妙味あり】期待値100%ライン突破！プラス収支が見込める狙い目です。**")
    elif prob >= 45.0:
        st.warning("⚖️ **【実力上位だが人気過剰】** 3着以内率は高いですが、オッズが低いため軸（連単・3連複の相手探し）向きです。")
    else:
        st.error("❄️ **【見送り推奨】** 確率・オッズともに妙味が薄いため、今回は見送りが無難です。")
