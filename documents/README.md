# 📚 Knowledge Base 知識庫

這個資料夾用來存放 chatbot 的知識庫檔案，透過 Gemini File Search 實現 RAG (Retrieval-Augmented Generation)。

## 📁 檔案說明

將您的知識文件（.md 檔案）放在這個資料夾中，然後執行上傳腳本，就能讓 chatbot 學習這些內容。

目前包含的檔案：
- `dc25s001.md` - dc25s017.md：糖尿病照護標準 (Diabetes Care Standards 2025)

## 🚀 使用方式

### 1️⃣ 上傳知識庫到 Gemini File Search

執行上傳腳本：

```bash
# 確保已設定 Google API Key
export GOOGLE_API_KEY='your_google_api_key_here'

# 執行上傳腳本
python upload_knowledge_base.py
```

### 2️⃣ 在 LINE Bot 中使用知識庫

上傳完成後，您的 LINE Bot 就能從這些文件中檢索資訊來回答問題。

例如：
```
👤 你: 糖尿病患者的 A1C 目標是多少？
🤖 Bot: 根據 2025 年糖尿病照護標準，大多數成年糖尿病患者
       的 A1C 目標是 <7%，但需要根據個人情況調整...
```

## 📝 新增文件

要新增更多知識文件：

1. 將 `.md` 檔案放入此資料夾
2. 執行 `python upload_knowledge_base.py`
3. 新文件會自動上傳到知識庫

## ⚠️ 注意事項

- **檔案格式**：建議使用 Markdown (.md) 格式
- **檔案大小**：注意單一檔案不要太大（建議 < 10MB）
- **macOS 用戶**：腳本會自動過濾 `.DS_Store` 和 `._*` 系統檔案
- **重複上傳**：重新執行腳本會上傳所有檔案（不會自動去重）

## 🔧 進階：自訂知識庫名稱

編輯 `upload_knowledge_base.py`，修改：

```python
KNOWLEDGE_BASE_STORE_NAME = "chatbot_knowledge_base"  # 改成您想要的名稱
```

## 🧹 清理 macOS 系統檔案

如果資料夾中出現 macOS 系統檔案，可以手動清理：

```bash
# 刪除 .DS_Store
find documents -name ".DS_Store" -type f -delete

# 刪除 ._* 檔案
find documents -name "._*" -type f -delete
```
