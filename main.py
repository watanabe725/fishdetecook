from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
from torchvision import transforms
from PIL import Image
from model import Net

from fastapi.middleware.cors import CORSMiddleware

# インスタンス化
app = FastAPI()

# CORSミドルウェアの設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Reactのフロントエンドを許可
    allow_credentials=True,
    allow_methods=["*"],  # すべてのHTTPメソッドを許可
    allow_headers=["*"],  # すべてのヘッダーを許可
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルをロードする場合
model = Net().to(device)  # 新しいインスタンスを作る

model.load_state_dict(torch.load('./model_FishDeteCook.pth', map_location=device))
model.eval()  # 推論モードにする

class_names = ["アジ", "タイ", "スズキ"]

# 2. 画像の前処理（例：画像分類用）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. 画像をロードして前処理
# image = Image.open("C:/Users/watan/kikagaku/fish_detecook/src/tai3.jpg")  # 予測対象の画像
# input_tensor = transform(image).unsqueeze(0).to(device)  # バッチ次元を追加

# トップページ
@app.get('/')
async def index():
    return {"Fish": 'fish_prediction'}
        
# POST が送信された時（入力）と予測値（出力）の定義
@app.post("/make_predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 画像をPIL形式で読み込む
        image = Image.open(file.file).convert("RGB")
        
        # 画像を前処理
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # 推論
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class = class_names[predicted.item()]

        return JSONResponse(content={"predicted_class": predicted_class})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))       