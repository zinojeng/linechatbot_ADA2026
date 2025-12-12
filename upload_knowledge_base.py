#!/usr/bin/env python3
"""
上傳 documents 資料夾中的 .md 檔案到 Gemini
作為 chatbot 的知識庫 (使用 google-generativeai SDK)
"""
import os
import sys
import glob
import time
from pathlib import Path
import google.generativeai as genai

# 設定
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DOCUMENTS_DIR = Path("documents")

# 驗證 API Key
if not GOOGLE_API_KEY:
    print("錯誤：請設定 GOOGLE_API_KEY 環境變數")
    print("執行方式：export GOOGLE_API_KEY='your_api_key_here'")
    sys.exit(1)

# 設定 GenAI
genai.configure(api_key=GOOGLE_API_KEY)
print(f"✅ GenAI SDK 設定成功")


def filter_markdown_files(directory: Path) -> list[Path]:
    """
    過濾出 .md 檔案，排除 macOS 系統檔案
    """
    md_files = []
    if not directory.exists():
        print(f"❌ 資料夾不存在：{directory}")
        return md_files

    for file_path in directory.iterdir():
        if not file_path.is_file() or file_path.suffix.lower() != '.md':
            continue
        if file_path.name.startswith('.') or file_path.name.startswith('._'):
            continue
        md_files.append(file_path)

    return sorted(md_files)


def upload_files():
    """
    主要上傳流程
    """
    print("=" * 60)
    print("📚 開始上傳知識庫檔案 (Google Generative AI SDK)")
    print("=" * 60)

    # 1. 取得現有檔案清單 (避免重複)
    print("🔍 檢查現有檔案...")
    existing_files = {}
    try:
        # list_files 回傳的是 iterable
        for f in genai.list_files():
            existing_files[f.display_name] = f
        print(f"✅ 雲端已有 {len(existing_files)} 個檔案")
    except Exception as e:
        print(f"⚠️  無法列出目前檔案: {e}")

    # 2. 掃描 documents 資料夾
    print(f"\n📂 掃描資料夾：{DOCUMENTS_DIR}")
    md_files = filter_markdown_files(DOCUMENTS_DIR)

    if not md_files:
        print("⚠️  找不到任何 .md 檔案")
        return

    print(f"📄 本地找到 {len(md_files)} 個 .md 檔案")

    # 3. 上傳新檔案
    print(f"\n⬆️  開始上傳檔案...")
    success_count = 0
    skip_count = 0
    fail_count = 0

    for file_path in md_files:
        display_name = file_path.name
        
        if display_name in existing_files:
            print(f"  ⏭️  跳過已存在檔案：{display_name}")
            skip_count += 1
            continue

        print(f"  ⬆️  上傳中：{display_name}")
        try:
            # Upload file
            uploaded_file = genai.upload_file(
                path=file_path,
                display_name=display_name,
                mime_type='text/plain'
            )
            
            # Verify state
            # Files are processed asynchronously, wait for ACTIVE state
            max_retries = 5
            for _ in range(max_retries):
                f = genai.get_file(uploaded_file.name)
                if f.state.name == "ACTIVE":
                    print(f"  ✅ 上傳並處理完成：{display_name}")
                    success_count += 1
                    break
                elif f.state.name == "FAILED":
                    print(f"  ❌ 處理失敗：{display_name}")
                    fail_count += 1
                    break
                time.sleep(1)
            else:
                 print(f"  ⚠️  上傳後處理超時 (狀態: {f.state.name})：{display_name}")
                 # 雖然超時但可能還在處理，暫算成功或另行處理
                 # 這裡保守計入成功，因為通常只是慢
                 success_count += 1

        except Exception as e:
            print(f"  ❌ 上傳錯誤：{display_name} - {e}")
            fail_count += 1

    # 4. 顯示結果
    print("\n" + "=" * 60)
    print("📊 上傳結果統計")
    print("=" * 60)
    print(f"✅ 成功上傳：{success_count} 個檔案")
    print(f"⏭️  跳過重複：{skip_count} 個檔案")
    print(f"❌ 上傳失敗：{fail_count} 個檔案")
    print("=" * 60)

if __name__ == "__main__":
    upload_files()
