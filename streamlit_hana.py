import streamlit as st
import requests
from PIL import Image
from io import BytesIO


# Streamlitアプリ
st.set_page_config(page_title="植物の名前は？", layout="centered")

# --- 💡 背景色のカスタムCSSを追加 ---
st.markdown(
    """
    <style>
    /* ページ全体の背景色をアリスブルーに */
    html, body, [data-testid="stApp"] {
        background-color: #fff9f4;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("🌿 植物の名前は？")
st.write("植物の画像をアップロードして、名前を調べましょう。")

# === アップロード方法の選択 ===
upload_method = st.radio("画像の入力方法を選んでください", ("ファイルをアップロード", "カメラで撮影"))


# # 画像アップロード
if "uploaded" not in st.session_state:
    st.session_state.uploaded = None

if upload_method == "ファイルをアップロード":
    file = st.file_uploader("画像をアップロード", type=["png", "jpg", "jpeg"])
    if file is not None:
        st.session_state.uploaded = file

elif upload_method == "カメラで撮影":
    image = st.camera_input("カメラで写真を撮ってください")
    if image is not None:
        st.session_state.uploaded = image

if st.session_state.uploaded is None:
    st.stop()

# --- アップロード画像で予測分析 ---
uploaded_file = st.session_state.uploaded 

st.success("🪴 画像が正常にアップロードされました！")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="アップロードされた画像", use_container_width=True)

    with st.spinner("🌺 画像を分類中..."):
        try:
            response = requests.post(
                "https://hana10.onrender.com/predict",  # ← ご自身のFastAPIサーバーURL
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                timeout=30  # タイムアウト長めに
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            st.error(f"❌ 予測APIに接続できませんでした。\n{e}")
            st.stop()

    if response.status_code == 200:
        results = response.json()["results"]
        st.subheader("🌺 予測結果")
        for i, result in enumerate(results, start=1):
            st.write(f"{i}. {result['name']}（予測率: {result['probability'] * 100:.2f}%）")

        top_flower = results[0]["name"]

        # 🌼 花の情報サイト検索リンクを作成　🌸
 
        # 🌼 タイトル行
        st.subheader(f"🔍 『{top_flower}』を各サイトで検索")
               
        # 🌿 サイト情報（アイコン, 公式URL, 表示名, 検索用URL）
        site_info = [
            ("🌿", "https://garden-vision.net", "花と緑の図鑑", f"https://www.google.com/search?q={top_flower}+site:garden-vision.net"),
            ("🌸", "https://www.shuminoengei.jp", "みんなの花図鑑", f"https://www.google.com/search?q={top_flower}+site:shuminoengei.jp/?m=pc&a=page_p_top"),
            ("🌼", "https://greensnap.jp", "GreenSnap", f"https://www.google.com/search?q={top_flower}+site:greensnap.jp"),
            ("🏠", "https://lovegreen.net", "LOVEGREEN", f"https://www.google.com/search?q={top_flower}+site:lovegreen.net/library"),
            ("🪴", "https://www.yasashi.info", "ヤサシイエンゲイ", f"https://www.google.com/search?q={top_flower}+site:yasashi.info"),
        ]

        # 🌱 サイトごとに並べる
        for emoji, official_url, site_name, search_url in site_info:
            left_col, right_col = st.columns([1, 9])
            with left_col:
                st.markdown(
                    f'<div style="display: inline-block; background-color: #f0f6da; padding: 3px 10px; border-radius: 10px; text-align: center; margin: 3px;">'
                    f'<a href="{official_url}" target="_blank" style="text-decoration: none;">'
                    f'<span style="font-size: 18px;">{emoji}</span>'
                    f'</a>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with right_col:
                st.markdown(
                    f'<div style="display: inline-block; background-color: #f0f6da; padding: 3px 10px; border-radius: 10px; text-align: left; margin: 3px;">'
                    f'<a href="{search_url}" target="_blank" style="text-decoration: none;">'
                    f'<span style="font-size: 18px;">🌐 {site_name}</span>'
                    f'</a>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    else:
        st.error("❌ 画像分類APIから予測結果を取得できませんでした。")
