#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict, deque
import threading
import cv2 
import numpy as np
import torch
from ultralytics import YOLO

class FrameReader(threading.Thread):
    def __init__(self, source, width=None, height=None, fps=None):
        super().__init__(daemon=True) # メインプログラム終了時に一緒に終了する設定
        self.cap = cv2.VideoCapture(source)
        
        # 【④ バッファサイズの最小化】（遅延をなくす魔法の1行）
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if width and height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            
        self.frame = None
        self.ret = False
        self.running = True
        self.lock = threading.Lock() # データの衝突を防ぐための鍵
        
        # 最初の1フレームを読み込んでおく
        if self.cap.isOpened():
            self.ret, self.frame = self.cap.read()

    def run(self):
        """別スレッドでひたすら最新フレームを読み込み続ける"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # lockを使って、AIが画像を読み取っている最中に上書きしないよう保護
                with self.lock:
                    self.ret = ret
                    self.frame = frame

    def read(self):
        """AI側（メインスレッド）から最新フレームを取得するメソッド"""
        with self.lock:
            # frame.copy() を返すことで、AI処理中に画像が書き換わるのを防ぐ
            return self.ret, (self.frame.copy() if self.frame is not None else None)

    def stop(self):
        """終了時の片付け"""
        self.running = False
        self.cap.release()

class WebcamPersonCounter:
    def __init__(self, source=0, model_path=None, line_position=0.5, line_direction='horizontal', 
                 conf_threshold=0.25, show_tracks=True, resolution=None, fps=None,
                 roi=(0.0, 0.0, 1.0, 1.0)):
        # 初期化パラメータ
        self.source = source
        self.model_path = model_path
        self.line_position = line_position
        self.line_direction = line_direction
        self.conf_threshold = conf_threshold
        self.show_tracks = show_tracks
        self.resolution = resolution
        self.fps = fps
        self.roi_norm = roi
        self.roi_pixels = None
        
        # カウンタの初期化
        self.count_up = 0
        self.count_down = 0
        self.counted_ids = set()
        self.crossed_ids = {}

        # 実測FPS計算用の変数
        self.prev_time = time.time()
        self.fps_display = 0.0
        
        # GPUが利用可能かチェック
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps') # MacのGPU(Metal)を指定
        else:
            self.device = torch.device('cpu')
            
        print(f"Using device: {self.device}")
        
        # YOLOモデルの読み込み
        if self.model_path is None:
            self.model = YOLO('yolov8n.pt', task='detect')
        else:
            self.model = YOLO(self.model_path, task='detect')
            
        print("AIモデルのウォームアップを開始します...")
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        # ダミー画像を1回だけ推論させる（結果は捨てる）
        self.model.track(dummy_img, persist=False, verbose=False, device=self.device, imgsz=(640, 640))
        print("ウォームアップ完了！")

        # 軌跡保存用のdequeだけ残す
        self.tracks = defaultdict(lambda: deque(maxlen=30))
        
        
        # カメラの初期化
        self.initialize_camera()
    
    def initialize_camera(self):
        """カメラを初期化し、別スレッドで起動する"""
        # 直列の VideoCapture の代わりに、さっき作った FrameReader を使う
        self.stream = FrameReader(self.source, self.resolution[0] if self.resolution else None, 
                                  self.resolution[1] if self.resolution else None, self.fps)
        if not self.stream.cap.isOpened():
            print(f"Error: カメラソース {self.source} を開けませんでした。")
            sys.exit(1)
        
        # スレッド（アルバイト）の稼働スタート！
        self.stream.start()
        
        # ※解像度とFPSの設定はFrameReaderの中で終わっているので、ここは「取得」だけでOK！
        # 画面サイズの取得（設定後の実際の値）
        self.width = int(self.stream.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.stream.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = self.stream.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"カメラ解像度: {self.width}x{self.height}")
        print(f"フレームレート: {self.actual_fps}")
        
        # ROIのピクセル座標を計算
        x1_n, y1_n, x2_n, y2_n = self.roi_norm
        self.roi_pixels = (
            int(self.width * x1_n),
            int(self.height * y1_n),
            int(self.width * x2_n),
            int(self.height * y2_n)
        )
        print(f"ROI（処理領域）: {self.roi_pixels}")
        
        # カウントラインの座標を計算
        if self.line_direction == 'horizontal':
            self.line_y = int(self.height * self.line_position)
            self.line_start = (0, self.line_y)
            self.line_end = (self.width, self.line_y)
        else:  # vertical
            self.line_x = int(self.width * self.line_position)
            self.line_start = (self.line_x, 0)
            self.line_end = (self.line_x, self.height)
    
    def list_available_cameras(self):
        """利用可能なカメラデバイスをリストアップする"""
        # クラス（PersonCounter）のメソッド(今は使ってない)
        available_cameras = []
        for i in range(10):  # 0から9までのカメラインデックスをチェック
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(i)
                cap.release()
        return available_cameras
    
    def is_wearing_orange(self, person_image, threshold=0.2):
        """指定された画像にオレンジ色が一定割合以上含まれているか判定する"""
        if person_image.size == 0:
            return False
    
        # BGRからHSV色空間に変換
        hsv_image = cv2.cvtColor(person_image, cv2.COLOR_BGR2HSV)

        # オレンジ色のHSV範囲を定義
        lower_orange = np.array([5, 100, 100])
        upper_orange = np.array([20, 255, 255])
        
        # マスクを作成
        mask = cv2.inRange(hsv_image, lower_orange, upper_orange)
        
        # オレンジ色のピクセル数を計算
        orange_pixels = cv2.countNonZero(mask)
        
        # 全ピクセル数を計算
        total_pixels = person_image.shape[0] * person_image.shape[1]
        if total_pixels == 0:
            return False
            
        # オレンジ色の割合を計算
        orange_ratio = orange_pixels / total_pixels
        
        # 割合が閾値を超えていればTrueを返す
        return orange_ratio > threshold
    
    def process_frame(self, frame):
        """フレームを処理して人物を検出・追跡し、カウントを更新"""

        # --- 1. 前処理（切り抜きなど） ---
        start_time = time.perf_counter()

        x1_roi, y1_roi, x2_roi, y2_roi = self.roi_pixels
        roi_frame = frame[y1_roi:y2_roi, x1_roi:x2_roi]

        pre_process_time = (time.perf_counter() - start_time) * 1000

        # --- 2. AI推論＆トラッキング ---
        start_time = time.perf_counter()

        if roi_frame.size == 0:
            print("警告: ROI領域が空です。ROIの指定を確認してください。")
            cv2.rectangle(frame, (x1_roi, y1_roi), (x2_roi, y2_roi), (255, 0, 0), 2)
            cv2.putText(frame, "ROI (Empty)", (x1_roi + 5, y1_roi + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            return frame

        # 【改善1】 half=True を追加してMac(MPS)での計算をさらに高速化
        results = self.model.track(roi_frame, persist=True, verbose=False, device=self.device, classes=[0], half=True, imgsz=(640, 640))[0]

        ai_time = (time.perf_counter() - start_time) * 1000

        # --- 3. 後処理＆描画 ---
        start_time = time.perf_counter()
        
        # 【改善2】ゴミ掃除用に「今画面にいるID」を記録するセットを用意
        current_active_ids = set()
            
        if results.boxes.id is not None:
            boxes_xyxy = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().tolist()
            confs = results.boxes.conf.cpu().numpy()
            # ※クラスID(cls)の取得は人間のみに絞ったため不要になり削除しました

            # 【改善3】二重ループを解消。1つのループで判定から描画まで一気に処理
            for i in range(len(track_ids)):
                conf = confs[i]
                
                # 信頼度が低い場合は、以後の計算を一切せずに次の人へ（処理の節約）
                if conf <= self.conf_threshold:
                    continue
                    
                bbox = boxes_xyxy[i]
                track_id = track_ids[i]
                
                # 座標を元のフレーム基準に戻す（描画時の高速化のため先に int にしておく）
                x1_orig = int(bbox[0] + x1_roi)
                y1_orig = int(bbox[1] + y1_roi)
                x2_orig = int(bbox[2] + x1_roi)
                y2_orig = int(bbox[3] + y1_roi)

                # 切り抜き時のエラーを防ぐため、画面の範囲内に座標を収める（クリッピング）
                x1_c = max(0, x1_orig)
                y1_c = max(0, y1_orig)
                x2_c = min(self.width, x2_orig)
                y2_c = min(self.height, y2_orig)

                # 元のフレームから人物領域だけを切り抜き
                person_crop = frame[y1_c:y2_c, x1_c:x2_c]

                # オレンジ色の服を着ていたら「弾く」（以降のカウントや描画処理をスキップ）
                if self.is_wearing_orange(person_crop):
                    continue

                # ゴミ掃除名簿に「この人は今いるよ」と記録
                current_active_ids.add(track_id)

                # 中心点の計算
                center_x = int((x1_orig + x2_orig) / 2)
                center_y = int((y1_orig + y2_orig) / 2)
                
                # 軌跡を更新
                self.tracks[track_id].append((center_x, center_y))
                track_history = self.tracks[track_id]
                
                # ラインの交差をチェック
                if len(track_history) >= 2:
                    prev_center = track_history[-2]
                    curr_center = track_history[-1]
                    
                    if self.line_direction == 'horizontal':
                        if (prev_center[1] < self.line_y and curr_center[1] >= self.line_y) or \
                           (prev_center[1] >= self.line_y and curr_center[1] < self.line_y):
                            if track_id not in self.crossed_ids:
                                if prev_center[1] < self.line_y and curr_center[1] >= self.line_y:
                                    self.count_down += 1
                                    self.crossed_ids[track_id] = 'down'
                                else:
                                    self.count_up += 1
                                    self.crossed_ids[track_id] = 'up'
                    else:  # vertical
                        if (prev_center[0] < self.line_x and curr_center[0] >= self.line_x) or \
                           (prev_center[0] >= self.line_x and curr_center[0] < self.line_x):
                            if track_id not in self.crossed_ids:
                                if prev_center[0] < self.line_x and curr_center[0] >= self.line_x:
                                    self.count_up += 1
                                    self.crossed_ids[track_id] = 'right'
                                else:
                                    self.count_down += 1
                                    self.crossed_ids[track_id] = 'left'
                
                # バウンディングボックスとIDを描画
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1_orig, y1_orig), (x2_orig, y2_orig), color, 2)
                cv2.putText(frame, f"ID: {track_id}", (x1_orig, y1_orig - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # 軌跡を描画
                if self.show_tracks and len(track_history) > 1:
                    # 過去の座標リスト(track_history)を、OpenCVが読めるNumPy配列(int32)に変換
                    pts = np.array(track_history, dtype=np.int32).reshape((-1, 1, 2))
                    # polylinesで一筆書き（isClosed=False で始点と終点を繋がないようにする）
                    cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2)

        # 【改善4】メモリのゴミ掃除（不要な軌跡データの削除）
        # self.tracks に記録されている全IDのうち、今画面にいない人の履歴フォルダを捨てる
        for track_id in list(self.tracks.keys()):
            if track_id not in current_active_ids:
                del self.tracks[track_id]
        
        # カウントラインを描画
        cv2.line(frame, self.line_start, self.line_end, (0, 0, 255), 2)
        
        # カウント情報を表示
        if self.line_direction == 'horizontal':
            cv2.putText(frame, f"Up: {self.count_up}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Down: {self.count_down}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:  # vertical
            cv2.putText(frame, f"Right: {self.count_up}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Left: {self.count_down}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Total: {self.count_up + self.count_down}", (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # カメラ情報とROI情報を表示
        cv2.putText(frame, f"Camera: {self.source} ({self.width}x{self.height} @ {self.actual_fps:.1f}fps)", 
                    (10, self.height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(frame, (x1_roi, y1_roi), (x2_roi, y2_roi), (255, 0, 0), 2)
        cv2.putText(frame, "ROI", (x1_roi + 5, y1_roi + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        draw_time = (time.perf_counter() - start_time) * 1000

        # ベンチマーク結果を表示
        cv2.putText(frame, f"AI Process: {ai_time:.1f} ms", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Draw/Other: {pre_process_time + draw_time:.1f} ms", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 実測FPSの計算（移動平均を使って数字のブレを抑える）
        current_time = time.time()
        elapsed_time = current_time - self.prev_time
        self.prev_time = current_time
        
        # 0除算エラーを防ぐ
        if elapsed_time > 0:
            current_fps = 1.0 / elapsed_time
            # 現在のFPSを10%、過去のFPSを90%の割合で混ぜて滑らかにする
            self.fps_display = (0.9 * self.fps_display) + (0.1 * current_fps)
            
        cv2.putText(frame, f"Actual FPS: {self.fps_display:.1f}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame
        
    def run(self):
        """メインループ"""
        while True:
            # テーブルに置かれた最新の画像をサッと取る（待ち時間ゼロ！）
            ret, frame = self.stream.read()
            
            if not ret or frame is None:
                continue # 画像がまだ来ていなければ一瞬待つ
            
            # フレームを処理
            processed_frame = self.process_frame(frame)
            
            # 結果を表示
            cv2.imshow("Webcam Person Counter", processed_frame)
            
            # キー入力を処理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # 'q'キーで終了
                break
            elif key == ord('r'):  # 'r'キーでカウントをリセット
                self.count_up = 0
                self.count_down = 0
                self.crossed_ids = {}
            elif key == ord('l'):  # 'l'キーでラインの方向を切り替え
                if self.line_direction == 'horizontal':
                    self.line_direction = 'vertical'
                    self.line_x = int(self.width * self.line_position)
                    self.line_start = (self.line_x, 0)
                    self.line_end = (self.line_x, self.height)
                else:
                    self.line_direction = 'horizontal'
                    self.line_y = int(self.height * self.line_position)
                    self.line_start = (0, self.line_y)
                    self.line_end = (self.width, self.line_y)
                print(f"ラインの方向を {self.line_direction} に変更しました")
        
        # リソースを解放
        self.stream.stop() # スレッドを安全に停止
        cv2.destroyAllWindows()


def list_cameras():
    """利用可能なカメラをリストアップして表示"""
    available_cameras = []
    for i in range(10):  # 0から9までのカメラインデックスをチェック
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                # カメラの情報を取得
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                available_cameras.append((i, width, height, fps))
            cap.release()
    
    if available_cameras:
        print("利用可能なカメラ:")
        for i, width, height, fps in available_cameras:
            print(f"  カメラID: {i}, 解像度: {width}x{height}, FPS: {fps:.1f}")
    else:
        print("利用可能なカメラが見つかりませんでした。")
    
    return available_cameras


def main():
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='Webカメラ人数カウントシステム')
    parser.add_argument('--source', type=int, default=0, help='カメラソース（デフォルト: 0）')
    parser.add_argument('--model', type=str, default=None, help='YOLOモデルのパス（デフォルト: yolo12x.pt）')
    parser.add_argument('--line-position', type=float, default=0.5, help='カウントラインの位置（0.0〜1.0、デフォルト: 0.5）')
    parser.add_argument('--line-direction', type=str, default='horizontal', choices=['horizontal', 'vertical'], 
                        help='ラインの方向（horizontal/vertical、デフォルト: horizontal）')
    parser.add_argument('--conf-threshold', type=float, default=0.25, help='信頼度閾値（デフォルト: 0.25）')
    parser.add_argument('--no-tracks', action='store_false', dest='show_tracks', help='軌跡を表示しない')
    parser.add_argument('--width', type=int, default=None, help='カメラの幅（ピクセル）')
    parser.add_argument('--height', type=int, default=None, help='カメラの高さ（ピクセル）')
    parser.add_argument('--fps', type=int, default=None, help='カメラのフレームレート')
    parser.add_argument('--list-cameras', action='store_true', help='利用可能なカメラをリストアップして終了')
    
    parser.add_argument('--roi-x1', type=float, default=0.0, help='ROIの左上X座標（割合 0.0-1.0、デフォルト: 0.0）')
    parser.add_argument('--roi-y1', type=float, default=0.0, help='ROIの左上Y座標（割合 0.0-1.0、デフォルト: 0.0）')
    parser.add_argument('--roi-x2', type=float, default=1.0, help='ROIの右下X座標（割合 0.0-1.0、デフォルト: 1.0）')
    parser.add_argument('--roi-y2', type=float, default=1.0, help='ROIの右下Y座標（割合 0.0-1.0、デフォルト: 1.0）')

    args = parser.parse_args()
    
    # カメラのリストアップ
    if args.list_cameras:
        list_cameras()
        return
    
    # 解像度の設定
    resolution = None
    if args.width is not None and args.height is not None:
        resolution = (args.width, args.height)
    
    # ROIの座標をタプルにまとめる
    roi = (args.roi_x1, args.roi_y1, args.roi_x2, args.roi_y2)

    # カウンターの初期化と実行
    counter = WebcamPersonCounter(
        source=args.source,
        model_path=args.model,
        line_position=args.line_position,
        line_direction=args.line_direction,
        conf_threshold=args.conf_threshold,
        show_tracks=args.show_tracks,
        resolution=resolution,
        fps=args.fps,
        roi=roi  # <-- ROI引数を渡す
    )
    
    print("\nキーボードショートカット:")
    print("  q: 終了")
    print("  r: カウントをリセット")
    print("  l: ラインの方向を切り替え（水平/垂直）")
    
    counter.run()


if __name__ == "__main__":
    main()