# main_monitor.py (日時表示機能付き)
import cv2
import time
import datetime
import os
import sys
from ultralytics import YOLO

# --- 1. 基本設定 ---
CAMERA_INDEX = 0          # USBカメラのインデックス
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
NATIVE_FPS = 30

# --- 2. 常時録画 (Low FPS) の設定 ---
LOW_FPS = 1.0
LOW_FPS_WRITE_INTERVAL = 1.0 / LOW_FPS
LOW_FPS_FILE_DURATION = 3 * 60 * 60
LOW_FPS_DIR = "CCTV/recordings_low"

# --- 3. イベント録画 (High FPS) の設定 ---
HIGH_FPS = 5.0
HIGH_FPS_WRITE_INTERVAL = 1.0 / HIGH_FPS
HIGH_FPS_DURATION = 20
HIGH_FPS_DIR = "CCTV/recordings_high"

# --- 4. YOLO (ultralytics) の設定 ---
COREML_MODEL_PATH = 'CCTV/yolov8n.mlpackage'
TARGET_CLASS_ID = 0  # 'person'

# --- 5. macOS用の録画設定 ---
FOURCC = cv2.VideoWriter_fourcc(*'mp4v')

# --- 6. フォルダ作成 ---
os.makedirs(LOW_FPS_DIR, exist_ok=True)
os.makedirs(HIGH_FPS_DIR, exist_ok=True)


def create_new_writer(directory, prefix, fps, width, height):
    """新しいVideoWriterを作成するヘルパー関数"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(directory, f"{prefix}_{timestamp}.mp4")
    print(f"[INFO] 新しい録画ファイルを作成: {filename} ({fps} FPS)")
    try:
        writer = cv2.VideoWriter(filename, FOURCC, fps, (width, height))
        if not writer.isOpened():
            raise IOError(f"VideoWriterが開けません: {filename}")
        return writer
    except Exception as e:
        print(f"[ERROR] VideoWriterの作成に失敗しました: {e}")
        return None

def run_yolo_ane(frame, model):
    """
    フレーム内でultralytics (Core ML/ANE) を実行し、
    人が検出されたか判定する
    """
    person_detected = False
    
    # ★★★ 信頼度の閾値を設定 ★★★
    CONF_THRESHOLD = 0.5 
    
    results = model.predict(
        frame,
        classes=[TARGET_CLASS_ID],
        conf=CONF_THRESHOLD,  # 閾値を指定
        verbose=False
    )
    
    result = results[0]

    if len(result.boxes) > 0:
        person_detected = True
        
        # 検出したボックスを描画
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            
            color = (0, 255, 0) # 緑
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"Person: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return person_detected, frame

# --- メイン処理 ---
def main():
    # --- 1. YOLOモデルの読み込み (ANE) ---
    print(f"[INFO] Neural Engine (ANE) 用のCore MLモデル ({COREML_MODEL_PATH}) を読み込みます...")
    try:
        global model
        model = YOLO(COREML_MODEL_PATH, task='detect') # 変更：モデルのタスクを明示
        print("[INFO] Core MLモデル読み込み完了。")
    except Exception as e:
        print(f"[ERROR] Core MLモデル '{COREML_MODEL_PATH}' の読み込みに失敗しました。")
        print(f"詳細: {e}")
        sys.exit()

    # --- 2. カメラの初期化 ---
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] カメラ {CAMERA_INDEX} を開けません。")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, NATIVE_FPS)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] カメラ起動。解像度: {actual_width}x{actual_height}")

    # --- 3. 録画ライターと状態変数の初期化 ---
    writer_low = create_new_writer(LOW_FPS_DIR, "low_fps", LOW_FPS, actual_width, actual_height)
    if writer_low is None: return
    
    low_writer_start_time = time.time()
    last_low_write_time = 0

    writer_high = None
    high_rec_end_time = 0
    last_high_write_time = 0
    
    detection_interval = 1.0 / HIGH_FPS
    last_detection_time = 0
    last_detection_result = False

    print("[INFO] 録画と監視を開始します。'q' キーで終了します。")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] フレームが読み込めません。")
                break
            
            # -------------------------------------------------
            # ★★★ 日時描画処理 (ここに追加) ★★★
            # -------------------------------------------------
            # 現在の日時を指定の形式で取得
            now_str = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            
            # 白い文字で左上に描画
            cv2.putText(frame,
                        now_str,
                        (10, 30),  # 描画位置 (X=10, Y=30)
                        cv2.FONT_HERSHEY_SIMPLEX,  # フォント
                        0.8,  # フォントサイズ
                        (255, 255, 255),  # 色 (白)
                        2,    # 太さ
                        cv2.LINE_AA) # アンチエイリアス
            # -------------------------------------------------

            current_time = time.time()
            
            # --- 1. 検出ロジック (5FPSで実行) ---
            if (current_time - last_detection_time) >= detection_interval:
                last_detection_time = current_time
                # 検出結果を'last_detection_result'に保存し、フレームに描画
                # (注意: 検出時にframeが上書きされるので、日時は先に描画しておく)
                last_detection_result, frame = run_yolo_ane(frame, model)
            
            # --- 2. イベント録画 (High FPS) ロジック ---
            if last_detection_result:
                if writer_high is None:
                    writer_high = create_new_writer(HIGH_FPS_DIR, "event", HIGH_FPS, actual_width, actual_height)
                
                high_rec_end_time = current_time + HIGH_FPS_DURATION
            
            if writer_high is not None:
                if current_time >= high_rec_end_time:
                    print("[INFO] イベント録画終了。")
                    writer_high.release()
                    writer_high = None
                elif (current_time - last_high_write_time) >= HIGH_FPS_WRITE_INTERVAL:
                    if writer_high.isOpened():
                        writer_high.write(frame)
                    last_high_write_time = current_time

            # --- 3. 常時録画 (Low FPS) ロジック ---
            if (current_time - last_low_write_time) >= LOW_FPS_WRITE_INTERVAL:
                if (current_time - low_writer_start_time) >= LOW_FPS_FILE_DURATION:
                    print("[INFO] 3時間が経過。常時録画ファイルをローテーションします。")
                    writer_low.release()
                    writer_low = create_new_writer(LOW_FPS_DIR, "low_fps", LOW_FPS, actual_width, actual_height)
                    if writer_low is None: break
                    low_writer_start_time = current_time
                
                if writer_low.isOpened():
                    writer_low.write(frame) # 日時が描画されたフレームが書き込まれる
                last_low_write_time = current_time

            # --- 4. 画面表示 (デバッグ用) ---
            cv2.imshow("Security Feed (Press 'q' to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # --- 終了処理 ---
        print("\n[INFO] 処理を終了します...")
        cap.release()
        if writer_low and writer_low.isOpened():
            writer_low.release()
        if writer_high and writer_high.isOpened():
            writer_high.release()
        cv2.destroyAllWindows()
        print("[INFO] 完了。")

if __name__ == "__main__":
    main()