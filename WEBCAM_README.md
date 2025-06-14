# Webカメラ人数カウントシステム

## 概要

このプログラムは、Webカメラを使用して人数をカウントするシステムです。YOLOv8とByteTrackを使用して人物検出と追跡を行い、指定したラインを通過する人数をリアルタイムでカウントします。

## 機能

- 複数のWebカメラから選択可能
- カメラの解像度とフレームレートの設定
- 水平/垂直のカウントライン
- リアルタイムの人数カウント（上/下、左/右）
- キーボードショートカットによる操作

## 必要条件

- Python 3.6以上
- OpenCV
- PyTorch
- Ultralytics YOLO
- NumPy

## インストール方法

1. 必要なパッケージをインストールします：

```bash
pip install opencv-python torch torchvision numpy ultralytics python-dotenv lapx
```

2. リポジトリをクローンまたはダウンロードします。

## 使用方法

### 利用可能なカメラの確認

```bash
python webcam_counter.py --list-cameras
```

### 基本的な使い方

```bash
python webcam_counter.py
```

これにより、デフォルトのカメラ（ID: 0）を使用して人数カウントが開始されます。

### カメラの選択

```bash
python webcam_counter.py --source 1
```

### 解像度とフレームレートの設定

```bash
python webcam_counter.py --width 1280 --height 720 --fps 30
```

### カウントラインの設定

```bash
python webcam_counter.py --line-position 0.7 --line-direction vertical
```

### その他のオプション

```bash
python webcam_counter.py --model yolov8s.pt --conf-threshold 0.3 --no-tracks
```

## キーボードショートカット

- `q`: プログラムを終了
- `r`: カウントをリセット
- `l`: ラインの方向を切り替え（水平/垂直）

## コマンドラインオプション

| オプション | 説明 |
|------------|------|
| `--source` | カメラソース（デフォルト: 0） |
| `--model` | YOLOモデルのパス（デフォルト: yolov8n.pt） |
| `--line-position` | カウントラインの位置（0.0〜1.0、デフォルト: 0.5） |
| `--line-direction` | ラインの方向（horizontal/vertical、デフォルト: horizontal） |
| `--conf-threshold` | 信頼度閾値（デフォルト: 0.25） |
| `--no-tracks` | 軌跡を表示しない |
| `--width` | カメラの幅（ピクセル） |
| `--height` | カメラの高さ（ピクセル） |
| `--fps` | カメラのフレームレート |
| `--list-cameras` | 利用可能なカメラをリストアップして終了 |

## 注意事項

- カメラへのアクセス権限が必要です
- GPUがある場合は自動的に使用されます
- 解像度やフレームレートはカメラの対応状況によって制限される場合があります