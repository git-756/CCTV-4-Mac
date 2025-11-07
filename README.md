# macOS CCTV - YOLOv8 & Discord通知

Apple Silicon (Mac) の Neural Engine (ANE) を活用し、Core ML に最適化された YOLOv8モデルを使用してリアルタイムで人物検出を行う防犯カメラ（CCTV）アプリケーションです。

人物を検出すると、自動的に高画質で録画を開始し、その動画クリップを即座にDiscordに通知します。

---

## ✨ 主な機能

- **Apple Core ML (ANE) 最適化**: `coremltools` を使用してエクスポートされた `.mlpackage` モデルを利用し、MacのGPU/ANEで高速な推論を実行します。
- **Discord通知**: 人物検出イベントが発生すると、録画されたMP4動画ファイルを指定のDiscordチャンネルに直接送信します。
- **デュアル録画モード**:
    - **常時録画 (低FPS)**: 常に低フレームレート（例: 1 FPS）で録画を継続し、一定時間（例: 3時間）ごとにファイルをローテーションします。
    - **イベント録画 (高FPS)**: 人物を検出した瞬間から、高フレームレート（例: 5 FPS）での録画を開始し、検出後も一定時間録画を継続します。
- **誤認識防止**: 影やノイズによる誤認識を減らすため、一定回数（例: 3回）連続で検出して初めてイベント録画を開始する「連続検出（Streak）」機能を搭載しています。

---

## ⚙️ 動作要件

- macOS (Apple Silicon Mac推奨)
- Python 3.8+
- `ultralytics`
- `torch` & `torchvision`
- `opencv-python`
- `coremltools`
- `discord-py`
- `python-dotenv`

---

## 🚀 使い方

1.  **リポジトリのクローンと依存関係のインストール**
    ```bash
    git clone [https://github.com/git-756/CCTV_mac.git](https://github.com/git-756/CCTV_mac.git)
    cd CCTV_mac
    # Ryeを使用している場合
    rye sync
    ```

2.  **YOLOモデルのCore MLエクスポート**
    - `ultralytics` がYOLOv8の `.pt` ファイルをダウンロードし、それを `.mlpackage` 形式に変換する必要があります。
    ```bash
    # (オプション) export_model.py を実行して yolov8n.mlpackage を生成
    rye run python CCTV/export_model.py
    ```

3.  **設定ファイルの作成**
    - `CCTV/.env.sample` をコピーし、`CCTV/.env` を作成します。
    - `.env` ファイルを開き、以下の情報を設定します。
        - `DISCORD_TOKEN`: あなたのDiscord Botのトークン
        - `DISCORD_CHANNEL_ID`: 通知を送信したいチャンネルのID
        - `COREML_MODEL_PATH`: `export_model.py` で生成した `.mlpackage` ファイルへのパス（例: `CCTV/yolov8n.mlpackage`）
        - （その他、録画設定やカメラ設定を必要に応じて調整します）

4.  **監視の実行**
    ```bash
    rye run python CCTV/main_monitor_discord.py
    ```
    （または `start_cctv.sh` を実行可能にして使用します）

---

## 📜 ライセンス

このプロジェクトは **MIT License** のもとで公開されています。ライセンスの全文については、[LICENSE](LICENSE) ファイルをご覧ください。

また、このプロジェクトはサードパーティ製のライブラリを利用しています。これらのライブラリ（`ultralytics`のAGPL-3.0ライセンス等）の情報については、[NOTICE.md](NOTICE.md) ファイルに記載しています。

## 作成者
[Samurai-Human-Go](https://samurai-human-go.com/%e9%81%8b%e5%96%b6%e8%80%85%e6%83%85%e5%a0%b1/)
- [【Mac M4 ANE】YOLOv8とPythonで監視カメラを自作！Discord通知と誤認識対策までの全開発記録](https://samurai-human-go.com/python-mac-m4-ane-cctv-yolo-discord/)