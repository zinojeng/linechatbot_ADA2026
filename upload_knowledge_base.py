#!/usr/bin/env python3
"""
上傳 documents 資料夾中的 .md 檔案到 Gemini File Search Store
作為 chatbot 的知識庫 (RAG - Retrieval-Augmented Generation)
"""
import os
import sys
import asyncio
from pathlib import Path
from google import genai
from google.genai import types

# 設定
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DOCUMENTS_DIR = Path("documents")
KNOWLEDGE_BASE_STORE_NAME = "chatbot_knowledge_base"  # 知識庫名稱

# 驗證 API Key
if not GOOGLE_API_KEY:
    print("錯誤：請設定 GOOGLE_API_KEY 環境變數")
    print("執行方式：export GOOGLE_API_KEY='your_api_key_here'")
    sys.exit(1)

# 初始化 GenAI client
client = genai.Client(api_key=GOOGLE_API_KEY)
print(f"✅ GenAI client 初始化成功")


def get_or_create_knowledge_base_store(store_display_name: str) -> str:
    """
    取得或建立知識庫 File Search Store
    Returns: actual_store_name (API 自動產生的名稱)
    """
    try:
        # 列出所有 stores，檢查是否已存在
        stores = client.file_search_stores.list()
        for store in stores:
            if hasattr(store, 'display_name') and store.display_name == store_display_name:
                print(f"📁 找到現有知識庫：{store.name}")
                return store.name

        # 不存在則建立新的
        print(f"📁 建立新知識庫：{store_display_name}")
        store = client.file_search_stores.create(
            config={'display_name': store_display_name}
        )
        print(f"✅ 知識庫建立成功：{store.name}")
        return store.name

    except Exception as e:
        print(f"❌ 建立知識庫時發生錯誤：{e}")
        sys.exit(1)


async def upload_file_to_store(file_path: Path, store_name: str, display_name: str) -> bool:
    """
    上傳單一檔案到 File Search Store
    """
    try:
        print(f"  ⬆️  上傳中：{display_name}")

        # 設定 MIME type（.md 檔案使用 text/plain）
        config_dict = {
            'display_name': display_name,
            'mime_type': 'text/plain'  # Markdown 檔案使用 text/plain
        }

        operation = client.file_search_stores.upload_to_file_search_store(
            file_search_store_name=store_name,
            file=str(file_path),
            config=config_dict
        )

        # 等待上傳完成
        max_wait = 60  # 秒
        elapsed = 0
        while not operation.done and elapsed < max_wait:
            await asyncio.sleep(2)
            operation = client.operations.get(operation)
            elapsed += 2

        if operation.done:
            print(f"  ✅ 上傳成功：{display_name}")
            return True
        else:
            print(f"  ⏱️  上傳超時：{display_name}")
            return False

    except Exception as e:
        print(f"  ❌ 上傳失敗：{display_name} - {e}")
        return False


def filter_markdown_files(directory: Path) -> list[Path]:
    """
    過濾出 .md 檔案，排除 macOS 系統檔案
    """
    md_files = []

    if not directory.exists():
        print(f"❌ 資料夾不存在：{directory}")
        return md_files

    for file_path in directory.iterdir():
        # 只處理 .md 檔案
        if not file_path.is_file() or file_path.suffix.lower() != '.md':
            continue

        # 排除 macOS 系統檔案
        if file_path.name.startswith('.') or file_path.name.startswith('._'):
            print(f"⏭️  跳過系統檔案：{file_path.name}")
            continue

        md_files.append(file_path)

    return sorted(md_files)


async def upload_knowledge_base():
    """
    主要上傳流程
    """
    print("=" * 60)
    print("📚 開始上傳知識庫檔案到 Gemini File Search")
    print("=" * 60)

    # 1. 取得或建立知識庫 store
    store_name = get_or_create_knowledge_base_store(KNOWLEDGE_BASE_STORE_NAME)

    # 2. 掃描 documents 資料夾
    print(f"\n📂 掃描資料夾：{DOCUMENTS_DIR}")
    md_files = filter_markdown_files(DOCUMENTS_DIR)

    if not md_files:
        print("⚠️  找不到任何 .md 檔案")
        return

    print(f"📄 找到 {len(md_files)} 個 .md 檔案")

    # 3. 上傳所有檔案
    print(f"\n⬆️  開始上傳檔案...")
    success_count = 0
    fail_count = 0

    for file_path in md_files:
        display_name = file_path.name
        success = await upload_file_to_store(file_path, store_name, display_name)

        if success:
            success_count += 1
        else:
            fail_count += 1

    # 4. 顯示結果
    print("\n" + "=" * 60)
    print("📊 上傳結果統計")
    print("=" * 60)
    print(f"✅ 成功：{success_count} 個檔案")
    print(f"❌ 失敗：{fail_count} 個檔案")
    print(f"📁 知識庫名稱：{KNOWLEDGE_BASE_STORE_NAME}")
    print(f"🔑 Store ID：{store_name}")
    print("\n💡 提示：現在您可以在 LINE Bot 中使用這些文件作為知識庫！")
    print("   只需在程式碼中查詢這個 store，Gemini 就能從這些文件中找答案。")
    print("=" * 60)


if __name__ == "__main__":
    # 執行上傳
    asyncio.run(upload_knowledge_base())
