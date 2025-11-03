# export_model.py
from ultralytics import YOLO
import sys

# 高速な 'nano' モデルを使用します
MODEL_NAME = 'yolov8n'
PT_FILE = f'{MODEL_NAME}.pt'
PACKAGE_FILE = f'{MODEL_NAME}.mlpackage'

print(f"[{MODEL_NAME}] モデルをロードしています...")
try:
    # .ptモデルをロード (自動でダウンロードされます)
    model = YOLO(PT_FILE)
except Exception as e:
    print(f"エラー: モデル '{PT_FILE}' のロードに失敗しました。 {e}")
    sys.exit()

print(f"[{MODEL_NAME}] をCore ML ({PACKAGE_FILE}) 形式にエクスポートします...")
try:
    # format='coreml' を指定してエクスポート
    model.export(format='coreml')
    print("\n---")
    print(f"✅ 成功: '{PACKAGE_FILE}' が作成されました。")
    print("---")
except Exception as e:
    print(f"\nエラー: Core MLへのエクスポートに失敗しました。 {e}")
    print("Rye環境で 'coremltools' がインストールされているか確認してください。")
