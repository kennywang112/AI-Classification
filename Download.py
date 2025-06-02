import gdown
import os

# 創建保存模型的目錄
model_dir = "./model/"
os.makedirs(model_dir, exist_ok=True)

# Google Drive 文件ID與文件名的對應
files = {
    "1yqn91aRCabUEfT7Ts7u7n4s75UHKPGlJ": "model.h5"
}
# 循環下載每個模型
for file_id, file_name in files.items():
    url = f"https://drive.google.com/uc?id={file_id}"
    output = os.path.join(model_dir, file_name)
    print(f"Downloading {file_name}...")
    gdown.download(url, output, quiet=False)
    print(f"{file_name} downloaded to {output}")