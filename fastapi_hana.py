from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import torch
from torchvision import transforms
from PIL import Image
import io
import pandas as pd
import os
import uvicorn


app = FastAPI()


# ===== Render起動：  ポート番号を取得して起動する方法=====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("fastapi_hana:app", host="0.0.0.0", port=10000)
    
# Render の Health Check    /health エンドポイントを作成する
@app.get("/health")
def health_check():
    return {"status": "ok"}


# # ===== トップページのメッセージ =====
from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
async def index():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>植物予測API</title>
        <style>
            body {
                font-family: 'Arial', sans-serif;
                background-color: #f0f8ff;
                color: #333;
                text-align: center;
                padding: 50px;
            }
            a {
                color: #007BFF;
                text-decoration: none;
                font-size: 18px;
            }
            a:hover {
                text-decoration: underline;
            }
            .box {
                background-color: #fff;
                border-radius: 10px;
                padding: 30px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                margin: 20px auto;
                max-width: 600px;
            }
        </style>
    </head>
    <body>
        <div class="box">
            <h1>🌿 植物名を予測するAPI 🌿</h1>
            <p>このAPIはアップロードした植物画像の名前を予測します</p>
            <p><a href="/docs" target="_blank">📄 植物名予測API ドキュメント</a></p>
        </div>
        <div class="box">
            <h1>🌿 植物名の予測サイト 🌿</h1>
            <p>植物画像をアップロードしてください</p>
            <p><a href="https://hana10-harunatsuakifuyu.streamlit.app/" target="_blank">📄 植物の名前は？</a></p>
        </div>
    </body>
    </html>
    """
    return html_content


# ===== CORSを許可（Streamlitとの通信のため）=====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== モデル定義（学習と同じNetクラス） =====
import torch.nn as nn
import torchvision.models as models

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.feature = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        h = self.feature(x)
        h = torch.flatten(h, 1)
        h = self.fc(h)
        return h


# ===== クラス名をCSVから読み込む　=====
class_names = pd.read_csv("class_names.csv", encoding="utf-8")["name"].tolist()

# ===== モデルの読み込み ===== 
model = Net()
checkpoint = torch.load("model_hana_with_fc.pt", map_location=torch.device("cpu"))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


# ===== 画像前処理　===== 
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# # ===== 予測エンドポイント（PORT 環境変数） =====
# import os
# port = int(os.environ.get("PORT", 8000))  # デフォルト8000、Render上ではPORTが与えられる

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=port)
    
# ===== 予測エンドポイント（推論用エンドポイント） =====
# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):  #str 型以外で使ってる場合に出ることがあります。
#         raise HTTPException(status_code=400, detail="画像ファイル（png/jpg/jpeg）をアップロードしてください。")
    
#     contents = await file.read()
#     try:
#         image = Image.open(io.BytesIO(contents)).convert("RGB")
#     except Exception:
#         raise HTTPException(status_code=400, detail="画像の読み込みに失敗しました。")
    
#     img_tensor = transform(image).unsqueeze(0)   # (1, 3, 224, 224)
#         # transform(image) の戻り値が Tensor じゃない場合に .unsqueeze(0) で失敗します。
#         # PIL → Tensor 変換が transform で失敗している場合、ファイルの中身が画像じゃない可能性もあります。
#         # ただし、コード上は .unsqueeze(0) の使い方に問題はありません。

#     with torch.no_grad():
#         outputs = model(img_tensor)
#         probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
#     top3_probs, top3_indices = torch.topk(probabilities, 3)
#     results = [
#         {"name": class_names[idx], "probability": float(prob)} # {"name": クラス名, "probability": 確率} 
#         for idx, prob in zip(top3_indices, top3_probs)
#     ]
#     return {"results": results}

# @app.post("/predict") をログ付きに変更

import logging

# # ログ設定
logging.basicConfig(level=logging.INFO)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    logging.info("✅ /predict にリクエストを受信")
    
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        logging.error("❌ 対応していないファイル形式")
        raise HTTPException(status_code=400, detail="画像ファイル（png/jpg/jpeg）をアップロードしてください。")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        logging.info("✅ 画像を読み込みました")
    except Exception as e:
        logging.error(f"❌ 画像読み込み失敗: {e}")
        raise HTTPException(status_code=400, detail="画像の読み込みに失敗しました。")

    try:
        img_tensor = transform(image).unsqueeze(0)
        logging.info(f"✅ 画像をテンソルに変換しました。形状: {img_tensor.shape}")
    except Exception as e:
        logging.error(f"❌ transform失敗: {e}")
        raise HTTPException(status_code=500, detail="画像の変換に失敗しました。")

    try:
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top3_probs, top3_indices = torch.topk(probabilities, 3)
            results = [
                {"name": class_names[idx], "probability": float(prob)}
                for idx, prob in zip(top3_indices, top3_probs)
            ]
        logging.info(f"✅ 推論成功: {results}")
    except Exception as e:
        logging.error(f"❌ 推論失敗: {e}")
        raise HTTPException(status_code=500, detail="推論処理に失敗しました。")

    return {"results": results}
