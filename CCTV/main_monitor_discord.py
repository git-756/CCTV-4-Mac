# main_monitor_discord.py
import cv2
import time
import datetime
import os
import sys
from ultralytics import YOLO
from dotenv import load_dotenv

# --- Discord連携のための追加インポート ---
import discord
import threading
import asyncio
# --------------------------------------

# --- 1. 基本設定 ---
# (変更なし)
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
NATIVE_FPS = 30

# --- 2. 常時録画 (Low FPS) の設定 ---
# (変更なし)
LOW_FPS = 1.0
LOW_FPS_WRITE_INTERVAL = 1.0 / LOW_FPS
LOW_FPS_FILE_DURATION = 3 * 60 * 60
LOW_FPS_DIR = "CCTV/recordings_low"

# --- 3. イベント録画 (High FPS) の設定 ---
# (変更なし)
HIGH_FPS = 5.0
HIGH_FPS_WRITE_INTERVAL = 1.0 / HIGH_FPS
HIGH_FPS_DURATION = 20
HIGH_FPS_DIR = "CCTV/recordings_high"

# --- 4. YOLO (ultralytics) の設定 ---
# (変更なし)
COREML_MODEL_PATH = 'CCTV/yolov8n.mlpackage'
TARGET_CLASS_ID = 0  # 'person'
CONF_THRESHOLD = 0.5 # 信頼度 (50%)

# --- 5. macOS用の録画設定 ---
# (変更なし)
FOURCC = cv2.VideoWriter_fourcc(*'mp4v')

# --- 6. フォルダ作成 ---
# (変更なし)
os.makedirs(LOW_FPS_DIR, exist_ok=True)
os.makedirs(HIGH_FPS_DIR, exist_ok=True)

# --- 7. Discord Bot 設定 ---
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
# チャンネルIDは整数(int)で取得
try:
    DISCORD_CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID"))
except (ValueError, TypeError):
    DISCORD_CHANNEL_ID = None

# Discordスレッドとメインスレッドの連携用
bot_loop = None
bot_client = None

# -----------------------------------------------
# --- Discord Bot 関連 (別スレッドで実行) ---
# -----------------------------------------------

def start_discord_bot():
    """Discord Botを起動し、別スレッドのイベントループで実行する"""
    global bot_loop, bot_client
    
    # 新しいイベントループをこのスレッド用に作成
    bot_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(bot_loop)

    # Botのクライアントを初期化 (ファイル送信権限のみ)
    intents = discord.Intents.default()
    bot_client = discord.Client(intents=intents)

    @bot_client.event
    async def on_ready():
        print(f"\n[INFO] Discord Botがバックグラウンドで起動しました。({bot_client.user})\n")

    try:
        # bot_loopでBotを起動 (メインスレッドをブロックしない)
        bot_loop.run_until_complete(bot_client.start(DISCORD_TOKEN))
    except discord.errors.LoginFailure:
        print("[ERROR] Discordトークンが無効です。")
    except Exception as e:
        print(f"[ERROR] Discord Botスレッドでエラー: {e}")

async def async_send_file(filepath):
    """(非同期) 実際のファイル送信処理"""
    if not bot_client or not DISCORD_CHANNEL_ID:
        print("[ERROR] Discord Botが初期化されていないか、チャンネルIDがありません。")
        return

    try:
        channel = bot_client.get_channel(DISCORD_CHANNEL_ID)
        if channel:
            print(f"[INFO] Discordへ動画を送信中... ({filepath})")
            now_str = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            await channel.send(
                f"🚨 **人検出イベント** ({now_str}) 🚨",
                file=discord.File(filepath)
            )
            print("[INFO] Discordへの送信完了。")
            
            # (オプション) 送信成功したらローカルのファイルを削除
            # os.remove(filepath) 
            # print(f"[INFO] ローカルファイル {filepath} を削除しました。")

        else:
            print(f"[ERROR] Discordチャンネル (ID: {DISCORD_CHANNEL_ID}) が見つかりません。")
    except discord.errors.Forbidden:
        print(f"[ERROR] Discord: チャンネル (ID: {DISCORD_CHANNEL_ID}) へのファイル送信権限がありません。")
    except Exception as e:
        print(f"[ERROR] Discordファイル送信中にエラー: {e}")

def send_discord_video(filepath):
    """(同期) メインスレッドから呼び出す関数"""
    if bot_loop and bot_client.is_ready():
        # メインスレッドから、Botスレッドのイベントループへタスクを投入
        asyncio.run_coroutine_threadsafe(
            async_send_file(filepath),
            bot_loop
        )
    else:
        print("[WARN] Discord Botがまだ準備できていないため、送信をスキップしました。")


# -----------------------------------------------
# --- OpenCV / YOLO 関連 (メインスレッド) ---
# -----------------------------------------------

def create_new_writer(directory, prefix, fps, width, height):
    """
    VideoWriterとファイル名を返すように変更
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(directory, f"{prefix}_{timestamp}.mp4")
    print(f"[INFO] 新しい録画ファイルを作成: {filename} ({fps} FPS)")
    try:
        writer = cv2.VideoWriter(filename, FOURCC, fps, (width, height))
        if not writer.isOpened():
            raise IOError(f"VideoWriterが開けません: {filename}")
        # writer と filename の両方を返す
        return writer, filename
    except Exception as e:
        print(f"[ERROR] VideoWriterの作成に失敗しました: {e}")
        return None, None

def run_yolo_ane(frame, model):
    """(変更なし) YOLO実行"""
    person_detected = False
    
    results = model.predict(
        frame,
        classes=[TARGET_CLASS_ID],
        conf=CONF_THRESHOLD,
        verbose=False
    )
    result = results[0]

    if len(result.boxes) > 0:
        person_detected = True
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            color = (0, 255, 0)
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
        model = YOLO(COREML_MODEL_PATH, task='detect') # task='detect' を明示
        print("[INFO] Core MLモデル読み込み完了。")
    except Exception as e:
        print(f"[ERROR] Core MLモデル '{COREML_MODEL_PATH}' の読み込みに失敗: {e}")
        sys.exit()

    # --- 2. Discord Botを別スレッドで起動 ---
    if not (DISCORD_TOKEN and DISCORD_CHANNEL_ID):
        print("[WARN] DISCORD_TOKEN または DISCORD_CHANNEL_ID が設定されていません。")
        print("[WARN] Discord通知機能は無効になります。")
    else:
        print("[INFO] Discord Botをバックグラウンドで起動します...")
        bot_thread = threading.Thread(target=start_discord_bot, daemon=True)
        bot_thread.start()
        # Botが起動するのを少し待つ
        time.sleep(5) 

    # --- 3. カメラの初期化 ---
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

    # --- 4. 録画ライターと状態変数の初期化 ---
    writer_low, _ = create_new_writer(LOW_FPS_DIR, "low_fps", LOW_FPS, actual_width, actual_height)
    if writer_low is None: return
    
    low_writer_start_time = time.time()
    last_low_write_time = 0

    writer_high = None
    high_filename = None # ★ イベント録画のファイル名を保持する変数
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
            
            # --- 日時描画 ---
            now_str = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            cv2.putText(frame, now_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, (255, 255, 255), 2, cv2.LINE_AA)

            current_time = time.time()
            
            # --- 1. 検出ロジック ---
            if (current_time - last_detection_time) >= detection_interval:
                last_detection_time = current_time
                last_detection_result, frame = run_yolo_ane(frame, model)
            
            # --- 2. イベント録画 (High FPS) ロジック ---
            if last_detection_result:
                if writer_high is None:
                    # ★ writer と filename を両方受け取る
                    writer_high, high_filename = create_new_writer(
                        HIGH_FPS_DIR, "event", HIGH_FPS, actual_width, actual_height
                    )
                
                high_rec_end_time = current_time + HIGH_FPS_DURATION
            
            if writer_high is not None:
                if current_time >= high_rec_end_time:
                    # ★ 録画終了 & Discord通知
                    print("[INFO] イベント録画終了。")
                    writer_high.release()
                    
                    # ★ Discord送信関数を呼び出す
                    if high_filename:
                        send_discord_video(high_filename)
                    
                    writer_high = None
                    high_filename = None # リセット

                elif (current_time - last_high_write_time) >= HIGH_FPS_WRITE_INTERVAL:
                    if writer_high.isOpened():
                        writer_high.write(frame)
                    last_high_write_time = current_time

            # --- 3. 常時録画 (Low FPS) ロジック ---
            if (current_time - last_low_write_time) >= LOW_FPS_WRITE_INTERVAL:
                if (current_time - low_writer_start_time) >= LOW_FPS_FILE_DURATION:
                    print("[INFO] 3時間が経過。常時録画ファイルをローテーションします。")
                    writer_low.release()
                    writer_low, _ = create_new_writer(
                        LOW_FPS_DIR, "low_fps", LOW_FPS, actual_width, actual_height
                    )
                    if writer_low is None: break
                    low_writer_start_time = current_time
                
                if writer_low.isOpened():
                    writer_low.write(frame)
                last_low_write_time = current_time

            # --- 4. 画面表示 ---
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
        
        # Botスレッドを安全に終了
        if bot_loop and bot_client:
            print("[INFO] Discord Botをシャットダウンします...")
            asyncio.run_coroutine_threadsafe(bot_client.close(), bot_loop)
        
        print("[INFO] 完了。")

if __name__ == "__main__":
    main()