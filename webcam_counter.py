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

# ByteTrackの簡易実装
class ByteTrack:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        # 初期化処理
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.id_count = 0
        self.tracks = defaultdict(lambda: deque(maxlen=30))  # 軌跡を保存

    def _iou(self, bbox1, bbox2):
        """IoU（Intersection over Union）を計算"""
        # bbox形式: [x1, y1, x2, y2]
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        return intersection / float(bbox1_area + bbox2_area - intersection)

    def update(self, detections):
        """検出結果を使用して追跡を更新"""
        self.frame_count += 1
        
        # 現在のトラッカーがない場合は新しいトラッカーを作成
        if len(self.trackers) == 0:
            for det in detections:
                self.trackers.append({
                    'id': self.id_count,
                    'bbox': det[:4],  # [x1, y1, x2, y2]
                    'conf': det[4],
                    'class': det[5],
                    'time_since_update': 0,
                    'hits': 1,
                    'age': 1
                })
                self.tracks[self.id_count].append((int((det[0] + det[2]) / 2), int((det[1] + det[3]) / 2)))
                self.id_count += 1
            return [t for t in self.trackers]
        
        # 既存のトラッカーと検出結果をマッチング
        matched, unmatched_dets, unmatched_trackers = [], [], []
        
        # 各トラッカーと検出結果のIoUを計算
        for t, tracker in enumerate(self.trackers):
            if tracker['time_since_update'] > self.max_age:
                unmatched_trackers.append(t)
                continue
                
            iou_max = self.iou_threshold
            m = -1
            
            for d, det in enumerate(detections):
                if d in matched:
                    continue
                    
                iou = self._iou(tracker['bbox'], det[:4])
                
                if iou > iou_max:
                    iou_max = iou
                    m = d
            
            if m != -1:
                matched.append(m)
                self.trackers[t]['bbox'] = detections[m][:4]
                self.trackers[t]['conf'] = detections[m][4]
                self.trackers[t]['time_since_update'] = 0
                self.trackers[t]['hits'] += 1
                self.trackers[t]['age'] += 1
                # 軌跡を更新
                center_x = int((detections[m][0] + detections[m][2]) / 2)
                center_y = int((detections[m][1] + detections[m][3]) / 2)
                self.tracks[self.trackers[t]['id']].append((center_x, center_y))
            else:
                unmatched_trackers.append(t)
        
        # 未マッチングの検出結果を新しいトラッカーとして追加
        for d in range(len(detections)):
            if d not in matched:
                unmatched_dets.append(d)
                
        for d in unmatched_dets:
            self.trackers.append({
                'id': self.id_count,
                'bbox': detections[d][:4],
                'conf': detections[d][4],
                'class': detections[d][5],
                'time_since_update': 0,
                'hits': 1,
                'age': 1
            })
            center_x = int((detections[d][0] + detections[d][2]) / 2)
            center_y = int((detections[d][1] + detections[d][3]) / 2)
            self.tracks[self.id_count].append((center_x, center_y))
            self.id_count += 1
        
        # 未マッチングのトラッカーを更新
        for t in unmatched_trackers:
            self.trackers[t]['time_since_update'] += 1
        
        # 一定期間更新されていないトラッカーを削除
        self.trackers = [t for t in self.trackers if t['time_since_update'] <= self.max_age]
        
        # 条件を満たすトラッカーのみ返す
        return [t for t in self.trackers if t['hits'] >= self.min_hits]


class WebcamPersonCounter:
    def __init__(self, source=0, model_path=None, line_position=0.5, line_direction='horizontal', 
                 conf_threshold=0.25, show_tracks=True, resolution=None, fps=None):
        # 初期化パラメータ
        self.source = source  # カメラソース（0=デフォルトカメラ）
        self.model_path = model_path  # YOLOモデルのパス（Noneの場合はデフォルトモデルを使用）
        self.line_position = line_position  # カウントラインの位置（0.0〜1.0）
        self.line_direction = line_direction  # ライン方向（'horizontal'または'vertical'）
        self.conf_threshold = conf_threshold  # 信頼度閾値
        self.show_tracks = show_tracks  # 軌跡を表示するかどうか
        self.resolution = resolution  # 解像度設定 (width, height)
        self.fps = fps  # フレームレート設定
        
        # カウンタの初期化
        self.count_up = 0  # 上/右方向のカウント
        self.count_down = 0  # 下/左方向のカウント
        self.counted_ids = set()  # すでにカウントしたIDを記録
        self.crossed_ids = {}  # ラインを横切ったIDと方向を記録
        
        # GPUが利用可能かチェック
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # YOLOモデルの読み込み
        if self.model_path is None:
            self.model = YOLO('yolov8n.pt')  # 小さいモデルを使用
        else:
            self.model = YOLO(self.model_path)
            
        # トラッカーの初期化
        self.tracker = ByteTrack()
        
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
        # YOLOで検出
        results = self.model(frame, verbose=False)[0]
        
        # 人物（クラス0）の検出結果のみをフィルタリング
        detections = []
        for r in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = r
            if int(cls) == 0 and conf > self.conf_threshold:  # クラス0=person
                detections.append([x1, y1, x2, y2, conf, cls])
        
        # トラッカーを更新
        tracks = self.tracker.update(detections)
        
        # 各トラックについて処理
        for track in tracks:
            track_id = track['id']
            bbox = track['bbox']
            
            # バウンディングボックスの中心点を計算
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)
            
            # 軌跡を取得
            track_history = self.tracker.tracks[track_id]
            
            # ラインの交差をチェック
            if len(track_history) >= 2:
                prev_center = track_history[-2]
                curr_center = track_history[-1]
                
                if self.line_direction == 'horizontal':
                    # 水平ラインの場合
                    if (prev_center[1] < self.line_y and curr_center[1] >= self.line_y) or \
                       (prev_center[1] >= self.line_y and curr_center[1] < self.line_y):
                        # まだカウントされていないIDの場合
                        if track_id not in self.crossed_ids:
                            # 上から下へ
                            if prev_center[1] < self.line_y and curr_center[1] >= self.line_y:
                                self.count_down += 1
                                self.crossed_ids[track_id] = 'down'
                            # 下から上へ
                            else:
                                self.count_up += 1
                                self.crossed_ids[track_id] = 'up'
                else:  # vertical
                    # 垂直ラインの場合
                    if (prev_center[0] < self.line_x and curr_center[0] >= self.line_x) or \
                       (prev_center[0] >= self.line_x and curr_center[0] < self.line_x):
                        # まだカウントされていないIDの場合
                        if track_id not in self.crossed_ids:
                            # 左から右へ
                            if prev_center[0] < self.line_x and curr_center[0] >= self.line_x:
                                self.count_up += 1
                                self.crossed_ids[track_id] = 'right'
                            # 右から左へ
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
    parser.add_argument('--model', type=str, default=None, help='YOLOモデルのパス（デフォルト: yolov8n.pt）')
    parser.add_argument('--line-position', type=float, default=0.5, help='カウントラインの位置（0.0〜1.0、デフォルト: 0.5）')
    parser.add_argument('--line-direction', type=str, default='horizontal', choices=['horizontal', 'vertical'], 
                        help='ラインの方向（horizontal/vertical、デフォルト: horizontal）')
    parser.add_argument('--conf-threshold', type=float, default=0.25, help='信頼度閾値（デフォルト: 0.25）')
    parser.add_argument('--no-tracks', action='store_false', dest='show_tracks', help='軌跡を表示しない')
    parser.add_argument('--width', type=int, default=None, help='カメラの幅（ピクセル）')
    parser.add_argument('--height', type=int, default=None, help='カメラの高さ（ピクセル）')
    parser.add_argument('--fps', type=int, default=None, help='カメラのフレームレート')
    parser.add_argument('--list-cameras', action='store_true', help='利用可能なカメラをリストアップして終了')
    
    args = parser.parse_args()
    
    # カメラのリストアップ
    if args.list_cameras:
        list_cameras()
        return
    
    # 解像度の設定
    resolution = None
    if args.width is not None and args.height is not None:
        resolution = (args.width, args.height)
    
    # カウンターの初期化と実行
    counter = WebcamPersonCounter(
        source=args.source,
        model_path=args.model,
        line_position=args.line_position,
        line_direction=args.line_direction,
        conf_threshold=args.conf_threshold,
        show_tracks=args.show_tracks,
        resolution=resolution,
        fps=args.fps
    )
    
    print("\nキーボードショートカット:")
    print("  q: 終了")
    print("  r: カウントをリセット")
    print("  l: ラインの方向を切り替え（水平/垂直）")
    
    counter.run()


if __name__ == "__main__":
    main()