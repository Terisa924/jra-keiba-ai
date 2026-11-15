import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="JRA競馬予想AI", page_icon="🏇", layout="wide")
st.title("🏇 マイJRA競馬予想AI（全頭一括ランキング対応版）")

# モデルとデータベースの読み込み（60秒ごとに更新チェック）
@st.cache_resource(ttl=60)
def load_all():
    model_v2 = joblib.load("jra_honkaku_model_v2.pkl")
    model_ana = joblib.load("jra_anauma_model.pkl")
    horse_db = pd.read_csv("horse_db.csv").set_index("馬名")
    jockey_db = pd.read_csv("jockey_db.csv").set_index("騎手")
    return model_v2, model_ana, horse_db, jockey_db

model_v2, model_ana, horse_db, jockey_db = load_all()

# モード選択（サイドバーまたは上部）
mode = st.radio(
    "使用するAIエンジンを選択してください",
    ["💎 穴馬発掘AI（実力のみ・期待値狙い）", "🔥 総合・的中重視AI（オッズ込み・軸馬探し）"],
    horizontal=True
)

# タブで「全頭一括モード」と「1頭詳細モード」を切り替え
tab1, tab2 = st.tabs(["🏆 全頭まとめて一括ランキング予想（最大18頭）", "🔍 1頭ずつ詳細分析（手動微調整可）"])

# ==========================================
# タブ1：全頭まとめて一括ランキング予想
# ==========================================
with tab1:
    st.subheader("1. レース共通条件を設定")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        b_place = st.selectbox("競馬場", ["東京", "中山", "京都", "阪神", "中京", "新潟", "福島", "小倉", "札幌", "函館"], key="b_place")
    with c2:
        b_surface = st.selectbox("芝・ダート", ["芝", "ダート"], key="b_surf")
    with c3:
        b_cond = st.selectbox("馬場状態", ["良", "稍重", "重", "不良"], key="b_cond")
    with c4:
        b_dist = st.number_input("距離 (m)", min_value=1000, max_value=3600, value=2000, step=100, key="b_dist")
    with c5:
        num_horses = st.number_input("出走頭数", min_value=2, max_value=18, value=16, step=1)

    st.subheader("2. 出馬表の入力（馬名・騎手を選ぶと過去成績は自動取得されます）")
    st.caption("※表のセルをクリックして編集できます。新馬やデータにない馬は「手入力馬」のままオッズ等を入れると標準値で計算します。")

    # 初期テーブルの作成（出走頭数分）
    horse_options = ["手入力馬"] + sorted(horse_db.index.tolist())
    jockey_options = ["標準騎手"] + sorted(jockey_db.index.tolist())

    init_rows = []
    for i in range(1, int(num_horses) + 1):
        waku_calc = min((i - 1) // 2 + 1, 8)
        init_rows.append({
            "枠番": waku_calc,
            "馬番": i,
            "馬名": "手入力馬",
            "騎手": "標準騎手",
            "斤量": 56.0,
            "単勝オッズ": 10.0,
            "人気": i,
            "馬体重増減": 0
        })
    df_entry = pd.DataFrame(init_rows)

    # Excelのように編集できるデータエディタ
    edited_df = st.data_editor(
        df_entry,
        column_config={
            "枠番": st.column_config.NumberColumn("枠番", min_value=1, max_value=8, step=1, width="small"),
            "馬番": st.column_config.NumberColumn("馬番", min_value=1, max_value=18, step=1, width="small"),
            "馬名": st.column_config.SelectboxColumn("馬名（検索可）", options=horse_options, required=True, width="medium"),
            "騎手": st.column_config.SelectboxColumn("騎手（検索可）", options=jockey_options, required=True, width="medium"),
            "斤量": st.column_config.NumberColumn("斤量", min_value=48.0, max_value=62.0, step=0.5, width="small"),
            "単勝オッズ": st.column_config.NumberColumn("単勝オッズ", min_value=1.0, max_value=999.9, step=0.1, width="small"),
            "人気": st.column_config.NumberColumn("人気", min_value=1, max_value=18, step=1, width="small"),
            "馬体重増減": st.column_config.NumberColumn("体重増減", min_value=-30, max_value=30, step=2, width="small"),
        },
        hide_index=True,
        use_container_width=True
    )

    if st.button("🏆 全頭まとめてAI予測＆ランキングを表示する！", type="primary", use_container_width=True):
        pred_rows = []
        for _, row in edited_df.iterrows():
            h_name = row["馬名"]
            j_name = row["騎手"]

            # 馬データベースから過去成績を自動取得
            if h_name != "手入力馬" and h_name in horse_db.index:
                h_info = horse_db.loc[h_name]
                sex = str(h_info["性別"]) if str(h_info["性別"]) in ["牡", "牝", "セ"] else "牡"
                age = int(min(max(h_info["前走時馬齢"], 2), 10))
                weight = float(h_info["前走馬体重"])
                prev_rank = float(h_info["前走着順"])
                prev_agari = float(h_info["前走上り3F"])
                prev_dist_diff = float(b_dist - h_info["前走距離"])
                avg_rank3 = float(h_info["過去3走平均着順"])
                db_status = "自動取得"
            else:
                sex, age, weight, prev_rank, prev_agari, prev_dist_diff, avg_rank3 = "牡", 4, 480.0, 5.0, 35.0, 0.0, 5.0
                db_status = "標準値"

            # 騎手データベースから複勝率を自動取得
            if j_name != "標準騎手" and j_name in jockey_db.index:
                j_rate = float(jockey_db.loc[j_name]["騎手_直近複勝率"])
            else:
                j_rate = 0.22

            pred_rows.append({
                '枠番': int(row["枠番"]),
                '馬番': int(row["馬番"]),
                '馬齢': age,
                '斤量': float(row["斤量"]),
                '距離(m)': int(b_dist),
                '単勝': float(row["単勝オッズ"]),
                '人気': int(row["人気"]),
                '馬体重': weight,
                '場体重増減': float(row["馬体重増減"]),
                '前走着順': prev_rank,
                '前走上り3F': prev_agari,
                '前走距離差': prev_dist_diff,
                '過去3走平均着順': avg_rank3,
                '騎手_直近複勝率': j_rate,
                '競馬場名': b_place,
                '芝・ダート区分': b_surface,
                '馬場状態1': b_cond,
                '性別': sex,
                '表示馬名': f"{int(row['馬番'])}番 {h_name}" if h_name != "手入力馬" else f"{int(row['馬番'])}番 (手入力馬)",
                '表示騎手': j_name,
                'データ状態': db_status
            })

        df_batch = pd.DataFrame(pred_rows)
        for col in ['競馬場名', '芝・ダート区分', '馬場状態1', '性別']:
            df_batch[col] = df_batch[col].astype('category')

        if "総合" in mode:
            features = ['枠番', '馬番', '馬齢', '斤量', '距離(m)', '単勝', '人気', '馬体重', '場体重増減',
                        '前走着順', '前走上り3F', '前走距離差', '過去3走平均着順', '競馬場名', '芝・ダート区分', '馬場状態1', '性別']
            probs = model_v2.predict_proba(df_batch[features])[:, 1] * 100
        else:
            features = ['枠番', '馬番', '馬齢', '斤量', '距離(m)', '馬体重', '場体重増減',
                        '前走着順', '前走上り3F', '前走距離差', '過去3走平均着順', '騎手_直近複勝率',
                        '競馬場名', '芝・ダート区分', '馬場状態1', '性別']
            probs = model_ana.predict_proba(df_batch[features])[:, 1] * 100

        df_batch['AI予測複勝率(%)'] = probs.round(1)
        # 単勝推定期待値(%) = (複勝率 * 0.35) * 単勝オッズ
        df_batch['単勝推定期待値(%)'] = ((probs * 0.35) * df_batch['単勝']).round(0).astype(int)

        # AI予測複勝率が高い順にランキング化
        df_result = df_batch.sort_values('AI予測複勝率(%)', ascending=False).reset_index(drop=True)
        df_result.index = df_result.index + 1
        df_result['AI順位'] = [f"🥇 1位" if i==1 else f"🥈 2位" if i==2 else f"🥉 3位" if i==3 else f"{i}位" for i in df_result.index]

        # 妙味判定マーク
        def judge_mark(r):
            if r['単勝推定期待値(%)'] >= 120:
                return "💰 超お宝(期待値120%超)"
            elif r['単勝推定期待値(%)'] >= 100:
                return "✅ 妙味あり(100%超)"
            elif r['AI予測複勝率(%)'] >= 40.0:
                return "⚖️ 軸候補(人気サイド)"
            else:
                return "－"

        df_result['AI判定'] = df_result.apply(judge_mark, axis=1)

        st.divider()
        st.subheader("🏅 AI全頭ランキング結果（予測複勝率順）")
        show_cols = ['AI順位', '枠番', '表示馬名', '表示騎手', '人気', '単勝', 'AI予測複勝率(%)', '単勝推定期待値(%)', 'AI判定', 'データ状態']
        st.dataframe(df_result[show_cols], use_container_width=True)

        # Geminiにそのままコピペできるテキストも自動生成！
        st.subheader("📋 Gemini（Gem）へのコピペ用分析データ")
        gem_text = f"【{b_place} {b_surface}{b_dist}m（馬場:{b_cond}）の自作AI全頭ランキング結果】\n"
        for _, r in df_result.iterrows():
            gem_text += f"{r['AI順位']}：{r['表示馬名']}（{r['表示騎手']} / {r['人気']}番人気 {r['単勝']}倍）→ AI複勝率:{r['AI予測複勝率(%)']}% / 期待値:{r['単勝推定期待値(%)']}% [{r['AI判定']}]\n"
        st.text_area("このままコピーしてGeminiに送ると展開・買い目を提案してくれます", gem_text, height=200)


# ==========================================
# タブ2：1頭ずつ詳細分析（従来の機能）
# ==========================================
with tab2:
    st.subheader("1. 馬名・騎手名の選択（過去データを自動取得）")
    col_search1, col_search2 = st.columns(2)
    with col_search1:
        horse_list = ["（新規・データ外の馬を手入力）"] + sorted(horse_db.index.tolist())
        selected_horse = st.selectbox("馬名を選択", horse_list, key="s_horse")
        if selected_horse == "（新規・データ外の馬を手入力）":
            horse_name = st.text_input("馬名を入力", "サンプルホース", key="s_hname")
            h_data = None
        else:
            horse_name = selected_horse
            h_data = horse_db.loc[selected_horse]
            st.caption(f"✅ 過去データ取得済（前走{int(h_data['前走着順'])}着 / 上り{h_data['前走上り3F']}秒）")

    with col_search2:
        jockey_list = ["（データ外の騎手・手動設定）"] + sorted(jockey_db.index.tolist())
        selected_jockey = st.selectbox("騎手を選択", jockey_list, key="s_jockey")
        if selected_jockey == "（データ外の騎手・手動設定）":
            default_j_rate = 25
        else:
            default_j_rate = int(jockey_db.loc[selected_jockey]['騎手_直近複勝率'] * 100)
            st.caption(f"✅ 騎手データ取得済（直近複勝率 {default_j_rate}%）")

    def_sex = str(h_data['性別']) if (h_data is not None and str(h_data['性別']) in ["牡", "牝", "セ"]) else "牝"
    def_age = int(min(max(h_data['前走時馬齢'], 2), 10)) if h_data is not None else 3
    def_weight = int(min(max(h_data['前走馬体重'], 380), 600)) if h_data is not None else 470
    def_prev_rank = int(min(max(h_data['前走着順'], 1), 18)) if h_data is not None else 3
    def_prev_agari = float(min(max(h_data['前走上り3F'], 31.0), 45.0)) if h_data is not None else 34.0
    def_prev_dist = int(h_data['前走距離']) if h_data is not None else 2000
    def_avg_rank3 = float(min(max(h_data['過去3走平均着順'], 1.0), 18.0)) if h_data is not None else 3.5

    col1, col2 = st.columns(2)
    with col1:
        place = st.selectbox("競馬場", ["東京", "中山", "京都", "阪神", "中京", "新潟", "福島", "小倉", "札幌", "函館"], key="s_place")
        surface = st.selectbox("芝・ダート", ["芝", "ダート"], key="s_surf")
        condition = st.selectbox("馬場状態", ["良", "稍重", "重", "不良"], key="s_cond")
        distance = st.number_input("今回の距離 (m)", min_value=1000, max_value=3600, value=2000, step=100, key="s_dist")

