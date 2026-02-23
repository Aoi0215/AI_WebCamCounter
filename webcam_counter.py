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
            self.model = YOLO('yolo12x.pt')  # 大きいモデルを使用
        else:
            self.model = YOLO(self.model_path)


        # 軌跡保存用のdequeだけ残す（公式トラッカーは軌跡を保持しないため）
        self.tracks = defaultdict(lambda: deque(maxlen=30))
        
        
        # カメラの初期化
        self.initialize_camera()
    
    def initialize_camera(self):
        """カメラを初期化する"""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            print(f"Error: カメラソース {self.source} を開けませんでした。")
            sys.exit(1)
        
        # 解像度の設定
        if self.resolution:
            width, height = self.resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # フレームレートの設定
        if self.fps:
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # 画面サイズの取得（設定後の実際の値）
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
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
    
    def process_frame(self, frame):
        """フレームを処理して人物を検出・追跡し、カウントを更新"""

        # 1. ROIのピクセル座標を取得
        x1_roi, y1_roi, x2_roi, y2_roi = self.roi_pixels
        
        # 2. フレームからROI領域を切り抜く
        roi_frame = frame[y1_roi:y2_roi, x1_roi:x2_roi]

        # 3. 切り抜いた画像 (roi_frame) をYOLOで検出
        if roi_frame.size == 0:
            print("警告: ROI領域が空です。ROIの指定を確認してください。")
            cv2.rectangle(frame, (x1_roi, y1_roi), (x2_roi, y2_roi), (255, 0, 0), 2)
            cv2.putText(frame, "ROI (Empty)", (x1_roi + 5, y1_roi + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            return frame

        # 3. 公式トラッカー(ByteTrack)を使い、切り抜いた画像を処理
        results = self.model.track(roi_frame, persist=True, verbose=False, device=self.device)[0]

        
        # 4. 検出結果の座標を、元のフレーム座標に変換
        detections = [] # [ (bbox), track_id ] のリスト
            
        # 検出と追跡が両方成功したかチェック
        if results.boxes.id is not None:
            
            # 必要なデータを個別にNumpy配列として取得
            boxes_xyxy = results.boxes.xyxy.cpu().numpy() # 座標 [x1, y1, x2, y2]
            track_ids = results.boxes.id.int().cpu().tolist() # トラックID
            confs = results.boxes.conf.cpu().numpy() # 信頼度
            clss = results.boxes.cls.cpu().numpy() # クラスID

            # 検出された数だけループ
            for i in range(len(track_ids)):
                bbox = boxes_xyxy[i]
                track_id = track_ids[i]
                conf = confs[i]
                cls = clss[i]
                
                # 座標を分解
                x1, y1, x2, y2 = bbox
                
                # オフセット (roi_pixelsの左上座標) を加算して、元の frame 基準の座標に戻す
                x1_orig = x1 + x1_roi
                y1_orig = y1 + y1_roi
                x2_orig = x2 + x1_roi
                y2_orig = y2 + y1_roi

                if int(cls) == 0 and conf > self.conf_threshold:  # クラス0=person
                    # (このスクリプトにはオレンジ色のフィルターはありません)
                    detections.append(([x1_orig, y1_orig, x2_orig, y2_orig], track_id))

        # 各トラックについて処理
        for (bbox, track_id) in detections:
            
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)
            
            # 軌跡を更新 (self.tracks を使う)
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
            
            # バウンディングボックスを描画
            color = (0, 255, 0)  # 緑色
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            
            # IDを表示
            cv2.putText(frame, f"ID: {track_id}", (int(bbox[0]), int(bbox[1])-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 軌跡を描画
            if self.show_tracks and len(track_history) > 1:
                for i in range(1, len(track_history)):
                    cv2.line(frame, track_history[i-1], track_history[i], color, 2)
        
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
        
        # カメラ情報を表示
        cv2.putText(frame, f"Camera: {self.source} ({self.width}x{self.height} @ {self.actual_fps:.1f}fps)", 
                    (10, self.height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # ROIの範囲を視覚化
        cv2.rectangle(frame, (x1_roi, y1_roi), (x2_roi, y2_roi), (255, 0, 0), 2) # 青色の枠
        cv2.putText(frame, "ROI", (x1_roi + 5, y1_roi + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return frame
    
    def run(self):
        """メインループ"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: フレームを読み込めませんでした。")
                break
            
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
        self.cap.release()
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