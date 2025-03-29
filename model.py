import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
from torchvision import transforms, datasets
import torchsummary
from torchsummary import summary
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, random_split
import torchvision.utils as vutils
from torchvision.models import resnet18, ResNet18_Weights
from torchmetrics.functional import accuracy
from net import Net
import pickle
from PIL import Image #Python Image Library


# 画像フォルダのパスを指定
image_folder_path = "C:/Users/watan/kikagaku/fish_detecook/data/fish_data"

# 学習済みモデルに合わせた前処理を追加
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ImageFolder を使ってデータセットを作成
dataset = datasets.ImageFolder(root=image_folder_path, transform=transform)

pl.seed_everything(0)

# 訓練・テストデータの分割（80% 訓練, 20% テスト）
train_val_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_val_size
val_size = int(0.2 * train_val_size)
train_size = train_val_size - val_size

train_val_dataset, test_dataset = random_split(dataset, [train_val_size, test_size])
train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [train_size, val_size])

# バッチサイズの定義
batch_size = 200

# Data Loader を定義
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size)

# # データの確認
# train_images, train_labels = next(iter(train_loader))
# test_images, test_labels = next(iter(test_loader))

# 確認
# print(f"画像データの形状: {train_images.shape}")  # (バッチサイズ, チャンネル, 高さ, 幅)
# print(f"ラベルの形状: {train_labels.shape}")  # (バッチサイズ,)
# print(f"ラベル: {train_labels.tolist()}")  # 数値ラベルをリストとして表示

# 画像の表示
# plt.figure(figsize=(10, 5))
# plt.axis("off")
# plt.title("Loaded Images")
# plt.imshow(vutils.make_grid(images, nrow=4).permute(1, 2, 0))  # grid にして表示
# plt.show()

if __name__ == "__main__":  # 直接実行されたときのみ学習を実行
    # 学習の実行
    pl.seed_everything(0)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = Net()

    logger = CSVLogger(save_dir='logs', name='my_exp')
    
    trainer = pl.Trainer(
        max_epochs=3, 
        accelerator=device, 
        devices=1,
        deterministic=False, 
        logger=logger)
    
    trainer.fit(net, train_loader, val_loader)

    # テストデータで検証
    results = trainer.test(dataloaders=test_loader)

    # モデルの重みのみを保存（推奨）
    torch.save(net.to(device).state_dict(), "model_FishDeteCook.pth")

    # # モデルをロードする場合
    # net = Net().to(device)  # 新しいインスタンスを作る
    # net.load_state_dict(torch.load('model_FishDeteCook.pth', map_location=device))
    # net.eval()  # 推論モードにする

    # # 画像をロードして前処理
    # image = Image.open("C:/Users/watan/kikagaku/fish_detecook/src/IMG_E2602.jpg")  # 推論したい画像
    # image = transform(image).unsqueeze(0).to(device)  # バッチ次元を追加

    # # 推論の実行
    # with torch.no_grad():  # 勾配計算を無効化（省メモリ＆高速化）
    #     output = net(image)

    # # 結果の処理（例: ソフトマックスで確率に変換）
    # probabilities = torch.nn.functional.softmax(output, dim=1)

    # # 最も確率の高いクラスを取得
    # predicted_class = torch.argmax(probabilities, dim=1).item()

    # print(f"予測クラス: {predicted_class}")