from fastapi import Request, FastAPI, HTTPException
import os
import sys
import asyncio
import aiohttp
import aiofiles
from pathlib import Path
from typing import Optional

from linebot.models import (
    MessageEvent, TextSendMessage, FileMessage, ImageMessage,
    PostbackEvent, TemplateSendMessage, CarouselTemplate, CarouselColumn,
    PostbackAction, FollowEvent
)
from linebot.exceptions import InvalidSignatureError
from linebot.aiohttp_async_http_client import AiohttpAsyncHttpClient
from linebot import AsyncLineBotApi, WebhookParser

# Google GenAI imports
from google import genai
from google.genai import types

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or ""

# LINE Bot configuration
channel_secret = os.getenv("ChannelSecret", None)
channel_access_token = os.getenv("ChannelAccessToken", None)

# Validate environment variables
if channel_secret is None:
    print("Specify ChannelSecret as environment variable.")
    sys.exit(1)
if channel_access_token is None:
    print("Specify ChannelAccessToken as environment variable.")
    sys.exit(1)
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY via env var or code.")

# Initialize GenAI client (Note: File Search API only supports Gemini API, not VertexAI)
client = genai.Client(api_key=GOOGLE_API_KEY)

print("GenAI client initialized successfully.")

# Initialize the FastAPI app for LINEBot
app = FastAPI()
client_session = aiohttp.ClientSession()
async_http_client = AiohttpAsyncHttpClient(client_session)
line_bot_api = AsyncLineBotApi(channel_access_token, async_http_client)
parser = WebhookParser(channel_secret)

# Create uploads directory if not exists
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Model configuration
MODEL_NAME = "gemini-2.5-flash"

# Knowledge Base configuration
KNOWLEDGE_BASE_STORE_NAME = "chatbot_knowledge_base"  # 共用知識庫名稱
USE_KNOWLEDGE_BASE = os.getenv("USE_KNOWLEDGE_BASE", "true").lower() == "true"  # 預設啟用知識庫

# User mode storage: {user_id: "personal" or "knowledge"}
user_modes = {}

# User profiles storage: {user_id: {profile_data}}
user_profiles = {}

# Onboarding state: {user_id: {"step": int, "data": {}}}
onboarding_state = {}

def get_user_id(event: MessageEvent) -> str:
    """取得使用者 ID"""
    return event.source.user_id


def get_user_mode(user_id: str) -> str:
    """
    取得使用者模式：'knowledge' 或 'personal'
    預設使用知識庫模式
    """
    return user_modes.get(user_id, "knowledge" if USE_KNOWLEDGE_BASE else "personal")


def set_user_mode(user_id: str, mode: str):
    """設定使用者模式"""
    user_modes[user_id] = mode


def get_user_profile(user_id: str) -> dict:
    """取得使用者資料"""
    return user_profiles.get(user_id, {})


def set_user_profile(user_id: str, profile: dict):
    """設定使用者資料"""
    user_profiles[user_id] = profile


def is_user_profile_complete(user_id: str) -> bool:
    """檢查使用者資料是否完整"""
    profile = get_user_profile(user_id)
    required_fields = ['name', 'age', 'gender', 'diabetes_type', 'education_level']
    return all(field in profile and profile[field] for field in required_fields)


def build_system_prompt(user_id: str) -> str:
    """
    根據使用者資料建立個人化的系統提示詞
    """
    profile = get_user_profile(user_id)

    if not profile:
        return ""

    # 基礎系統提示
    prompt_parts = ["請根據以下患者資訊提供個人化的衛教內容：\n"]

    # 加入使用者基本資訊
    if profile.get('name'):
        prompt_parts.append(f"• 患者稱呼：{profile['name']}")

    if profile.get('age'):
        age = profile['age']
        prompt_parts.append(f"• 年齡：{age}歲")
        # 根據年齡調整說明方式
        if int(age) < 18:
            prompt_parts.append("  → 使用適合青少年理解的簡單語言")
        elif int(age) >= 65:
            prompt_parts.append("  → 特別注意老年人的用藥安全和低血糖風險")

    if profile.get('gender'):
        prompt_parts.append(f"• 性別：{profile['gender']}")
        if profile['gender'] == '女性':
            prompt_parts.append("  → 考慮妊娠糖尿病和更年期影響")

    if profile.get('diabetes_type'):
        dtype = profile['diabetes_type']
        prompt_parts.append(f"• 糖尿病類型：{dtype}")
        if dtype == '第1型':
            prompt_parts.append("  → 強調胰島素治療的重要性")
        elif dtype == '第2型':
            prompt_parts.append("  → 著重生活方式調整和口服藥物")
        elif dtype == '妊娠糖尿病':
            prompt_parts.append("  → 關注母嬰健康和產後追蹤")

    if profile.get('complications'):
        prompt_parts.append(f"• 併發症：{', '.join(profile['complications'])}")
        prompt_parts.append("  → 針對現有併發症提供預防惡化的建議")

    if profile.get('education_level'):
        edu = profile['education_level']
        prompt_parts.append(f"• 教育程度：{edu}")
        if edu in ['國小', '國中']:
            prompt_parts.append("  → 使用淺顯易懂的詞彙，避免醫學術語")
        elif edu in ['大學', '研究所']:
            prompt_parts.append("  → 可以使用較專業的醫學詞彙和詳細解釋")

    if profile.get('current_medications'):
        prompt_parts.append(f"• 目前用藥：{', '.join(profile['current_medications'])}")
        prompt_parts.append("  → 注意藥物交互作用和副作用")

    # 回答風格指引
    prompt_parts.append("\n【回答原則】")
    prompt_parts.append("1. 使用溫和、支持性的語氣")
    prompt_parts.append("2. 根據患者的教育程度調整專業術語的使用")
    prompt_parts.append("3. 提供具體、可執行的建議")
    prompt_parts.append("4. 強調個人化照護的重要性")
    prompt_parts.append("5. 必要時建議諮詢醫療專業人員\n")

    return "\n".join(prompt_parts)


def get_store_name(event: MessageEvent) -> str:
    """
    根據使用者模式和訊息來源，取得 file search store 名稱
    - knowledge 模式：使用共用知識庫
    - personal 模式：使用個人/群組文件庫
    """
    user_id = get_user_id(event)
    mode = get_user_mode(user_id)

    # 知識庫模式：使用共用知識庫
    if mode == "knowledge":
        return KNOWLEDGE_BASE_STORE_NAME

    # 個人模式：根據來源類型決定
    if event.source.type == "user":
        return f"user_{event.source.user_id}"
    elif event.source.type == "group":
        return f"group_{event.source.group_id}"
    elif event.source.type == "room":
        return f"room_{event.source.room_id}"
    else:
        return f"unknown_{event.source.user_id}"


async def download_line_content(message_id: str, file_name: str) -> Optional[Path]:
    """
    Download file content from LINE and save to local uploads directory.
    Returns the local file path if successful, None otherwise.
    """
    try:
        # Get message content from LINE
        message_content = await line_bot_api.get_message_content(message_id)

        # Extract file extension from original file name
        _, ext = os.path.splitext(file_name)
        # Use safe file name (ASCII only) to avoid encoding issues
        safe_file_name = f"{message_id}{ext}"
        file_path = UPLOAD_DIR / safe_file_name

        async with aiofiles.open(file_path, 'wb') as f:
            async for chunk in message_content.iter_content():
                await f.write(chunk)

        print(f"Downloaded file: {file_path} (original: {file_name})")
        return file_path
    except Exception as e:
        print(f"Error downloading file: {e}")
        return None


async def ensure_file_search_store_exists(store_name: str) -> tuple[bool, str]:
    """
    Ensure file search store exists, create if not.
    Returns (success, actual_store_name).
    Note: store_name is used as display_name, but actual name is auto-generated by API.
    """
    try:
        # List all stores and check if one with our display_name exists
        stores = client.file_search_stores.list()
        for store in stores:
            if hasattr(store, 'display_name') and store.display_name == store_name:
                print(f"File search store '{store_name}' already exists: {store.name}")
                return True, store.name

        # Store doesn't exist, create it
        print(f"Creating file search store with display_name '{store_name}'...")
        store = client.file_search_stores.create(
            config={'display_name': store_name}
        )
        print(f"File search store created: {store.name} (display_name: {store_name})")
        return True, store.name

    except Exception as e:
        print(f"Error ensuring file search store exists: {e}")
        return False, ""


# Cache to store display_name -> actual_name mapping
store_name_cache = {}


async def list_documents_in_store(store_name: str) -> list:
    """
    List all documents in a file search store.
    Returns list of document info dicts.
    """
    try:
        # Get actual store name
        actual_store_name = None
        if store_name in store_name_cache:
            actual_store_name = store_name_cache[store_name]
        else:
            # Find store by display_name
            stores = client.file_search_stores.list()
            for store in stores:
                if hasattr(store, 'display_name') and store.display_name == store_name:
                    actual_store_name = store.name
                    store_name_cache[store_name] = actual_store_name
                    break

        if not actual_store_name:
            print(f"Store '{store_name}' not found")
            return []

        documents = []

        # Try to use SDK method first
        if hasattr(client.file_search_stores, 'documents'):
            for doc in client.file_search_stores.documents.list(parent=actual_store_name):
                documents.append({
                    'name': doc.name,
                    'display_name': getattr(doc, 'display_name', 'Unknown'),
                    'create_time': str(getattr(doc, 'create_time', '')),
                    'update_time': str(getattr(doc, 'update_time', ''))
                })
                print(f"Use SDK list function: File found in store '{store_name}': {doc.name}")
        else:
            # Fallback to REST API
            import requests
            url = f"https://generativelanguage.googleapis.com/v1beta/{actual_store_name}/documents"
            headers = {'Content-Type': 'application/json'}
            params = {'key': GOOGLE_API_KEY}

            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            for doc in data.get('documents', []):
                documents.append({
                    'name': doc.get('name', 'N/A'),
                    'display_name': doc.get('displayName', 'Unknown'),
                    'create_time': doc.get('createTime', ''),
                    'update_time': doc.get('updateTime', '')
                })
                print(f"Use REST API list function: File found in store '{store_name}': {doc.name}")
        return documents

    except Exception as e:
        print(f"Error listing documents in store: {e}")
        return []


async def delete_document(document_name: str) -> bool:
    """
    Delete a document from file search store.
    Returns True if successful, False otherwise.
    Note: force=True is required to permanently delete documents from File Search Store.
    """
    try:
        # Try to use SDK method first with force=True
        try:
            if hasattr(client.file_search_stores, 'documents'):
                # Force delete is required for File Search Store documents
                client.file_search_stores.documents.delete(
                    name=document_name,
                    config={'force': True}
                )
                print(f"Document deleted successfully with force=True: {document_name}")
                return True
        except Exception as sdk_error:
            print(f"SDK delete failed, trying REST API: {sdk_error}")

        # Fallback to REST API with force parameter
        import requests
        url = f"https://generativelanguage.googleapis.com/v1beta/{document_name}"
        headers = {'Content-Type': 'application/json'}
        params = {
            'key': GOOGLE_API_KEY,
            'force': 'true'  # Required for File Search Store documents
        }

        response = requests.delete(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()

        print(f"Document deleted successfully via REST API with force=true: {document_name}")
        return True

    except Exception as e:
        print(f"Error deleting document: {e}")
        return False


async def upload_to_file_search_store(file_path: Path, store_name: str, display_name: Optional[str] = None) -> bool:
    """
    Upload a file to Gemini file search store.
    Returns True if successful, False otherwise.
    """
    try:
        # Check cache first
        if store_name in store_name_cache:
            actual_store_name = store_name_cache[store_name]
            print(f"Using cached store name: {actual_store_name}")
        else:
            # Ensure the store exists before uploading
            success, actual_store_name = await ensure_file_search_store_exists(store_name)
            if not success:
                print(f"Failed to ensure store '{store_name}' exists")
                return False
            # Cache the mapping
            store_name_cache[store_name] = actual_store_name

        # Upload to file search store
        # actual_store_name is the API-generated name (e.g., fileSearchStores/xxx)
        # display_name is the custom display name for the file (used in citations)
        config_dict = {}
        if display_name:
            config_dict['display_name'] = display_name

        operation = client.file_search_stores.upload_to_file_search_store(
            file_search_store_name=actual_store_name,
            file=str(file_path),
            config=config_dict if config_dict else None
        )

        # Wait for operation to complete (with timeout)
        max_wait = 60  # seconds
        elapsed = 0
        while not operation.done and elapsed < max_wait:
            await asyncio.sleep(2)
            operation = client.operations.get(operation)
            elapsed += 2

        if operation.done:
            print(f"File uploaded to store '{store_name}': {operation}")
            return True
        else:
            print(f"Upload operation timeout for store '{store_name}'")
            return False

    except Exception as e:
        print(f"Error uploading to file search store: {e}")
        return False


def clean_markdown(text: str) -> str:
    """
    移除 Markdown 格式符號，讓訊息在 LINE 中更易讀
    """
    import re

    # 移除粗體標記 **text** 或 __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)

    # 移除斜體標記 *text* 或 _text_
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'_(.+?)_', r'\1', text)

    # 移除刪除線 ~~text~~
    text = re.sub(r'~~(.+?)~~', r'\1', text)

    # 移除行內程式碼標記 `text`
    text = re.sub(r'`(.+?)`', r'\1', text)

    # 移除標題符號 # ## ### 等
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    # 移除連結 [text](url) 保留文字
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)

    # 移除圖片 ![alt](url)
    text = re.sub(r'!\[.+?\]\(.+?\)', '', text)

    # 移除程式碼區塊標記 ```
    text = re.sub(r'```[\w]*\n', '', text)
    text = re.sub(r'```', '', text)

    # 移除引用標記 >
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)

    # 移除水平線 --- 或 ***
    text = re.sub(r'^(\*{3,}|-{3,}|_{3,})$', '', text, flags=re.MULTILINE)

    # 清理列表項目符號（保留結構但美化）
    # 無序列表 - * +
    text = re.sub(r'^\s*[-*+]\s+', '• ', text, flags=re.MULTILINE)

    # 有序列表 1. 2. 3.
    # 保持原樣，因為數字編號在 LINE 中也很清楚

    # 移除多餘空行（超過兩個連續空行）
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 移除前後空白
    text = text.strip()

    return text


async def query_file_search(query: str, store_name: str, user_id: str = None) -> str:
    """
    Query the file search store using generate_content with personalized context.
    Returns the AI response text (cleaned from markdown).
    """
    try:
        # Get actual store name from cache or by searching
        actual_store_name = None

        if store_name in store_name_cache:
            actual_store_name = store_name_cache[store_name]
            print(f"Using cached store name for query: {actual_store_name}")
        else:
            # Try to find the store by display_name
            try:
                stores = client.file_search_stores.list()
                for store in stores:
                    if hasattr(store, 'display_name') and store.display_name == store_name:
                        actual_store_name = store.name
                        store_name_cache[store_name] = actual_store_name
                        print(f"Found store for query: {actual_store_name}")
                        break
            except Exception as list_error:
                print(f"Error listing stores: {list_error}")

        if not actual_store_name:
            # Store doesn't exist - guide user to upload files
            print(f"File search store '{store_name}' not found")
            return "📁 您還沒有上傳任何檔案。\n\n請先傳送文件檔案（PDF、DOCX、TXT 等）給我，上傳完成後就可以開始提問了！\n\n💡 提示：如果您想分析圖片，請直接傳送圖片給我，我會立即為您分析。"

        # Create FileSearch tool with actual store name
        tool = types.Tool(
            file_search=types.FileSearch(
                file_search_store_names=[actual_store_name]
            )
        )

        # 建立個人化系統提示詞
        system_prompt = ""
        if user_id:
            system_prompt = build_system_prompt(user_id)

        # 組合查詢內容
        if system_prompt:
            # 如果有個人化提示，將其加入查詢
            full_query = f"{system_prompt}\n\n【患者問題】\n{query}"
        else:
            full_query = query

        # Generate content with file search
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=full_query,
            config=types.GenerateContentConfig(
                tools=[tool],
                temperature=0.7,
            )
        )

        # Extract text from response
        if response.text:
            # 清理 Markdown 符號，讓訊息在 LINE 中更易讀
            cleaned_text = clean_markdown(response.text)
            return cleaned_text
        else:
            return "抱歉，我無法從文件中找到相關資訊。"

    except Exception as e:
        print(f"Error querying file search: {e}")
        # Check if error is related to missing store
        if "not found" in str(e).lower() or "does not exist" in str(e).lower():
            return "📁 您還沒有上傳任何檔案。\n\n請先傳送文件檔案（PDF、DOCX、TXT 等）給我，上傳完成後就可以開始提問了！\n\n💡 提示：如果您想分析圖片，請直接傳送圖片給我，我會立即為您分析。"
        return f"查詢時發生錯誤：{str(e)}"


async def analyze_image_with_gemini(image_path: Path) -> str:
    """
    Analyze image using Gemini's vision capability.
    Returns the analysis result text.
    """
    try:
        # Read image bytes
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        # Determine MIME type based on file extension
        ext = image_path.suffix.lower()
        mime_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        mime_type = mime_type_map.get(ext, 'image/jpeg')

        # Create image part
        image = types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type
        )

        # Generate content with image
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=["請詳細描述這張圖片的內容，包括主要物品、場景、文字等資訊。", image],
        )

        if response.text:
            # 清理 Markdown 符號
            cleaned_text = clean_markdown(response.text)
            return cleaned_text
        else:
            return "抱歉，我無法分析這張圖片。"

    except Exception as e:
        print(f"Error analyzing image with Gemini: {e}")
        return f"圖片分析時發生錯誤：{str(e)}"


async def handle_image_message(event: MessageEvent, message: ImageMessage):
    """
    Handle image messages - analyze using Gemini vision.
    """
    file_name = f"image_{message.id}.jpg"

    # Download image
    reply_msg = TextSendMessage(text="正在分析您的圖片，請稍候...")
    await line_bot_api.reply_message(event.reply_token, reply_msg)

    file_path = await download_line_content(message.id, file_name)

    if file_path is None:
        error_msg = TextSendMessage(text="圖片下載失敗，請重試。")
        await line_bot_api.push_message(event.source.user_id, error_msg)
        return

    # Analyze image with Gemini
    analysis_result = await analyze_image_with_gemini(file_path)

    # Clean up local file
    try:
        file_path.unlink()
    except Exception as e:
        print(f"Error deleting file: {e}")

    # Send analysis result
    result_msg = TextSendMessage(text=f"📸 圖片分析結果：\n\n{analysis_result}")
    await line_bot_api.push_message(event.source.user_id, result_msg)


async def handle_document_message(event: MessageEvent, message: FileMessage):
    """
    Handle file messages - download and upload to file search store.
    """
    store_name = get_store_name(event)
    file_name = message.file_name or "unknown_file"

    # Download file
    reply_msg = TextSendMessage(text="正在處理您的檔案，請稍候...")
    await line_bot_api.reply_message(event.reply_token, reply_msg)

    file_path = await download_line_content(message.id, file_name)

    if file_path is None:
        error_msg = TextSendMessage(text="檔案下載失敗，請重試。")
        await line_bot_api.push_message(event.source.user_id, error_msg)
        return

    # Upload to file search store
    success = await upload_to_file_search_store(file_path, store_name, file_name)

    # Clean up local file
    try:
        file_path.unlink()
    except Exception as e:
        print(f"Error deleting file: {e}")

    if success:
        success_msg = TextSendMessage(
            text=f"✅ 檔案已成功上傳！\n檔案名稱：{file_name}\n\n現在您可以詢問我關於這個檔案的任何問題。"
        )
        await line_bot_api.push_message(event.source.user_id, success_msg)
    else:
        error_msg = TextSendMessage(text="檔案上傳失敗，請重試。")
        await line_bot_api.push_message(event.source.user_id, error_msg)


def is_list_files_intent(text: str) -> bool:
    """
    Check if user wants to list files.
    """
    list_keywords = [
        '列出檔案', '列出文件', '顯示檔案', '顯示文件',
        '查看檔案', '查看文件', '檔案列表', '文件列表',
        '有哪些檔案', '有哪些文件', '我的檔案', '我的文件',
        'list files', 'show files', 'my files'
    ]
    text_lower = text.lower().strip()
    return any(keyword in text_lower for keyword in list_keywords)


def is_mode_switch_intent(text: str) -> tuple[bool, str]:
    """
    檢查使用者是否想切換模式
    Returns: (is_switch, mode) - mode 可為 'knowledge' 或 'personal'
    """
    text_lower = text.lower().strip()

    # 知識庫模式關鍵字
    knowledge_keywords = [
        '知識庫', '知識庫模式', '使用知識庫', '切換知識庫',
        '糖尿病', '醫療知識', '專業知識',
        'knowledge', 'knowledge base'
    ]

    # 個人模式關鍵字
    personal_keywords = [
        '個人檔案', '個人模式', '我的檔案', '私人檔案',
        '切換個人', '使用個人',
        'personal', 'my files', 'personal mode'
    ]

    # 檢查是否包含「模式」或「切換」等字眼
    is_mode_command = any(word in text_lower for word in ['模式', '切換', 'mode', 'switch'])

    if is_mode_command:
        if any(kw in text_lower for kw in knowledge_keywords):
            return True, 'knowledge'
        elif any(kw in text_lower for kw in personal_keywords):
            return True, 'personal'

    return False, ''


def get_mode_description(mode: str) -> str:
    """取得模式說明"""
    if mode == 'knowledge':
        return "📚 知識庫模式\n使用共用醫療知識庫（糖尿病照護標準 2025）回答問題"
    else:
        return "📁 個人模式\n使用您上傳的個人文件回答問題"


def start_onboarding(user_id: str):
    """開始使用者資料收集流程"""
    onboarding_state[user_id] = {
        "step": 1,
        "data": {}
    }


def get_onboarding_question(step: int) -> str:
    """取得 onboarding 問題"""
    questions = {
        1: "👋 您好！為了提供更個人化的衛教建議，請問我該如何稱呼您？\n（例如：王先生、小美、張媽媽）",
        2: "請問您的年齡是？\n（請輸入數字，例如：45）",
        3: "請問您的性別是？\n（請輸入：男性 或 女性）",
        4: "請問您的糖尿病類型是？\n請選擇：\n1. 第1型糖尿病\n2. 第2型糖尿病\n3. 妊娠糖尿病\n4. 其他類型",
        5: "請問您目前有以下併發症嗎？（可複選，用逗號分隔）\n1. 視網膜病變\n2. 腎臟病變\n3. 神經病變\n4. 心血管疾病\n5. 足部病變\n6. 無\n\n例如：1,3 或 6",
        6: "請問您的教育程度是？\n1. 國小\n2. 國中\n3. 高中/職\n4. 大學\n5. 研究所",
        7: "最後一個問題：您目前有在使用哪些藥物嗎？（可選填）\n（請輸入藥名，用逗號分隔，或輸入「無」）"
    }
    return questions.get(step, "")


async def process_onboarding_answer(user_id: str, answer: str) -> str:
    """處理 onboarding 回答"""
    state = onboarding_state.get(user_id)
    if not state:
        return None

    step = state["step"]
    data = state["data"]

    # 處理不同步驟的回答
    if step == 1:  # 稱呼
        data['name'] = answer.strip()
        state["step"] = 2
        return f"很高興認識您，{data['name']}！\n\n{get_onboarding_question(2)}"

    elif step == 2:  # 年齡
        try:
            age = int(answer.strip())
            if age < 0 or age > 120:
                return "請輸入有效的年齡（0-120）"
            data['age'] = str(age)
            state["step"] = 3
            return get_onboarding_question(3)
        except:
            return "請輸入有效的數字年齡"

    elif step == 3:  # 性別
        gender = answer.strip()
        if gender in ['男性', '女性', '男', '女']:
            data['gender'] = '男性' if gender in ['男性', '男'] else '女性'
            state["step"] = 4
            return get_onboarding_question(4)
        else:
            return "請輸入「男性」或「女性」"

    elif step == 4:  # 糖尿病類型
        type_map = {
            '1': '第1型',
            '2': '第2型',
            '3': '妊娠糖尿病',
            '4': '其他'
        }
        dtype = type_map.get(answer.strip())
        if dtype:
            data['diabetes_type'] = dtype
            state["step"] = 5
            return get_onboarding_question(5)
        else:
            return "請輸入 1、2、3 或 4"

    elif step == 5:  # 併發症
        comp_map = {
            '1': '視網膜病變',
            '2': '腎臟病變',
            '3': '神經病變',
            '4': '心血管疾病',
            '5': '足部病變'
        }
        answer = answer.strip()
        if answer == '6' or answer.lower() == '無':
            data['complications'] = []
        else:
            try:
                selected = [comp_map[num.strip()] for num in answer.split(',') if num.strip() in comp_map]
                data['complications'] = selected
            except:
                return "請輸入有效的選項（例如：1,3 或 6）"
        state["step"] = 6
        return get_onboarding_question(6)

    elif step == 6:  # 教育程度
        edu_map = {
            '1': '國小',
            '2': '國中',
            '3': '高中/職',
            '4': '大學',
            '5': '研究所'
        }
        edu = edu_map.get(answer.strip())
        if edu:
            data['education_level'] = edu
            state["step"] = 7
            return get_onboarding_question(7)
        else:
            return "請輸入 1、2、3、4 或 5"

    elif step == 7:  # 藥物（最後一步）
        answer = answer.strip()
        if answer.lower() in ['無', 'none', '']:
            data['current_medications'] = []
        else:
            medications = [med.strip() for med in answer.split(',')]
            data['current_medications'] = medications

        # 完成 onboarding，儲存資料
        set_user_profile(user_id, data)
        del onboarding_state[user_id]

        # 建立完成訊息
        summary = f"""
✅ 資料建立完成！

【您的個人資料】
• 稱呼：{data['name']}
• 年齡：{data['age']}歲
• 性別：{data['gender']}
• 糖尿病類型：{data['diabetes_type']}
• 併發症：{', '.join(data['complications']) if data['complications'] else '無'}
• 教育程度：{data['education_level']}
• 目前用藥：{', '.join(data['current_medications']) if data['current_medications'] else '無'}

現在我會根據您的個人資料提供更適合您的衛教建議！

您可以隨時輸入「我的資料」查看或「更新資料」重新設定。

現在就開始提問吧！ 😊
"""
        return summary

    return None


def is_onboarding_command(text: str) -> bool:
    """檢查是否為開始設定資料的指令"""
    keywords = ['設定資料', '建立資料', '個人資料設定', '開始設定', 'setup', 'start']
    return any(keyword in text.lower() for keyword in keywords)


def is_profile_view_command(text: str) -> bool:
    """檢查是否為查看資料的指令"""
    keywords = ['我的資料', '個人資料', '查看資料', 'my profile', 'profile']
    return any(keyword in text.lower() for keyword in keywords)


def is_profile_update_command(text: str) -> bool:
    """檢查是否為更新資料的指令"""
    keywords = ['更新資料', '修改資料', '重新設定', 'update profile']
    return any(keyword in text.lower() for keyword in keywords)


async def send_files_carousel(event: MessageEvent, documents: list):
    """
    Send files as LINE Carousel Template.
    """
    if not documents:
        no_files_msg = TextSendMessage(text="📁 目前沒有任何文件。\n\n請先上傳文件檔案，就可以查詢囉！")
        await line_bot_api.reply_message(event.reply_token, no_files_msg)
        return

    # LINE Carousel限制最多10個
    documents = documents[:10]

    columns = []
    for doc in documents:
        # 提取檔名（去除路徑部分）
        display_name = doc.get('display_name', 'Unknown')
        # 格式化時間
        create_time = doc.get('create_time', '')
        if create_time and 'T' in create_time:
            # 簡化時間顯示 (YYYY-MM-DD HH:MM)
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(create_time.replace('Z', '+00:00'))
                create_time = dt.strftime('%Y-%m-%d %H:%M')
            except:
                create_time = create_time[:16]  # 簡單截斷

        # 建立每個檔案的 Column
        column = CarouselColumn(
            thumbnail_image_url='https://via.placeholder.com/1024x1024/4CAF50/FFFFFF?text=File',  # 預設圖片
            title=display_name[:40],  # LINE 限制標題長度
            text=f"上傳時間：{create_time[:20]}" if create_time else "文件檔案",
            actions=[
                PostbackAction(
                    label='🗑️ 刪除檔案',
                    data=f"action=delete_file&doc_name={doc['name']}"
                )
            ]
        )
        columns.append(column)

    carousel_template = CarouselTemplate(columns=columns)
    template_message = TemplateSendMessage(
        alt_text=f'📁 找到 {len(documents)} 個文件',
        template=carousel_template
    )

    await line_bot_api.reply_message(event.reply_token, template_message)


async def handle_postback(event: PostbackEvent):
    """
    Handle postback events (e.g., delete file button clicks).
    """
    try:
        # Parse postback data
        data = event.postback.data
        params = dict(param.split('=') for param in data.split('&'))

        action = params.get('action')
        doc_name = params.get('doc_name')

        if action == 'delete_file' and doc_name:
            # Delete the document
            success = await delete_document(doc_name)

            if success:
                # Extract display name from doc_name for user-friendly message
                display_name = doc_name.split('/')[-1] if '/' in doc_name else doc_name

                reply_msg = TextSendMessage(
                    text=f"✅ 檔案已刪除成功！\n\n如需查看剩餘檔案，請輸入「列出檔案」。"
                )
            else:
                reply_msg = TextSendMessage(text="❌ 刪除檔案失敗，請稍後再試。")

            await line_bot_api.reply_message(event.reply_token, reply_msg)
        else:
            print(f"Unknown postback action: {action}")

    except Exception as e:
        print(f"Error handling postback: {e}")
        error_msg = TextSendMessage(text="處理操作時發生錯誤。")
        await line_bot_api.reply_message(event.reply_token, error_msg)


async def handle_follow_event(event: FollowEvent):
    """
    Handle follow event - when user first adds the bot or re-adds after blocking.
    Automatically start onboarding process.
    """
    user_id = event.source.user_id

    print(f"New user followed: {user_id}")

    # Check if user already has a complete profile
    if is_user_profile_complete(user_id):
        # User already has profile, just send welcome back message
        welcome_msg = TextSendMessage(
            text="👋 歡迎回來！\n\n我是您的糖尿病照護助手。\n\n您可以：\n• 詢問糖尿病相關問題（使用知識庫模式）\n• 上傳個人文件進行查詢（切換個人模式）\n• 輸入「我的資料」查看個人資料\n\n現在就開始提問吧！😊"
        )
        await line_bot_api.reply_message(event.reply_token, welcome_msg)
    else:
        # New user or incomplete profile, start onboarding
        start_onboarding(user_id)
        welcome_text = f"""👋 您好！歡迎使用糖尿病照護助手！

我可以幫助您：
📚 解答糖尿病相關問題
💊 提供用藥與照護建議
📊 分析您上傳的健康文件
🖼️ 解讀醫療影像與報告

為了提供更個人化的衛教內容，讓我先了解您的基本資料。

{get_onboarding_question(1)}"""

        welcome_msg = TextSendMessage(text=welcome_text)
        await line_bot_api.reply_message(event.reply_token, welcome_msg)


async def handle_text_message(event: MessageEvent, message):
    """
    Handle text messages - onboarding, switch mode, list files, or query with personalization.
    """
    user_id = get_user_id(event)
    query = message.text
    current_mode = get_user_mode(user_id)

    print(f"Received query: {query} from user: {user_id}, mode: {current_mode}")

    # 0. Check if user is in onboarding process
    if user_id in onboarding_state:
        response = await process_onboarding_answer(user_id, query)
        reply_msg = TextSendMessage(text=response)
        await line_bot_api.reply_message(event.reply_token, reply_msg)
        return

    # 1. Check if user wants to start onboarding
    if is_onboarding_command(query):
        start_onboarding(user_id)
        reply_msg = TextSendMessage(text=get_onboarding_question(1))
        await line_bot_api.reply_message(event.reply_token, reply_msg)
        return

    # 2. Check if user wants to view their profile
    if is_profile_view_command(query):
        profile = get_user_profile(user_id)
        if profile:
            profile_text = f"""
📋 【您的個人資料】

• 稱呼：{profile.get('name', '未設定')}
• 年齡：{profile.get('age', '未設定')}歲
• 性別：{profile.get('gender', '未設定')}
• 糖尿病類型：{profile.get('diabetes_type', '未設定')}
• 併發症：{', '.join(profile.get('complications', [])) if profile.get('complications') else '無'}
• 教育程度：{profile.get('education_level', '未設定')}
• 目前用藥：{', '.join(profile.get('current_medications', [])) if profile.get('current_medications') else '無'}

💡 輸入「更新資料」可重新設定
"""
        else:
            profile_text = "您還沒有設定個人資料。\n\n輸入「設定資料」開始建立個人化衛教檔案。"
        reply_msg = TextSendMessage(text=profile_text)
        await line_bot_api.reply_message(event.reply_token, reply_msg)
        return

    # 3. Check if user wants to update their profile
    if is_profile_update_command(query):
        start_onboarding(user_id)
        reply_msg = TextSendMessage(text="♻️ 重新設定個人資料\n\n" + get_onboarding_question(1))
        await line_bot_api.reply_message(event.reply_token, reply_msg)
        return

    # 4. Check if user wants to switch mode
    is_switch, new_mode = is_mode_switch_intent(query)
    if is_switch:
        set_user_mode(user_id, new_mode)
        mode_desc = get_mode_description(new_mode)
        reply_text = f"✅ 已切換到 {mode_desc}\n\n現在可以開始提問了！"
        reply_msg = TextSendMessage(text=reply_text)
        await line_bot_api.reply_message(event.reply_token, reply_msg)
        return

    # 5. Check if user wants to see current mode
    if query.strip() in ['模式', '目前模式', '當前模式', 'mode', 'current mode']:
        mode_desc = get_mode_description(current_mode)
        profile_status = "✅ 已設定" if is_user_profile_complete(user_id) else "❌ 未設定"
        reply_text = f"🔍 目前模式：\n\n{mode_desc}\n\n👤 個人資料：{profile_status}\n\n💡 切換模式請輸入：\n• 「切換知識庫模式」\n• 「切換個人模式」\n\n💡 個人化衛教請輸入：\n• 「設定資料」"
        reply_msg = TextSendMessage(text=reply_text)
        await line_bot_api.reply_message(event.reply_token, reply_msg)
        return

    # 6. Get store name based on current mode
    store_name = get_store_name(event)

    # 7. Check if user wants to list files
    if is_list_files_intent(query):
        documents = await list_documents_in_store(store_name)
        await send_files_carousel(event, documents)
        return

    # 8. Otherwise, query file search with personalization
    # Check if user has complete profile for personalized response
    has_profile = is_user_profile_complete(user_id)

    response_text = await query_file_search(query, store_name, user_id)

    # Add mode indicator to response
    mode_indicator = "📚" if current_mode == "knowledge" else "📁"
    response_with_mode = f"{mode_indicator} {response_text}"

    # Add friendly reminder if user doesn't have complete profile
    if not has_profile and current_mode == "knowledge":
        response_with_mode += "\n\n💡 提示：設定個人資料後，我可以根據您的年齡、教育程度、糖尿病類型等提供更適合您的建議。\n\n輸入「設定資料」開始個人化設定。"

    # Reply to user
    reply_msg = TextSendMessage(text=response_with_mode)
    await line_bot_api.reply_message(event.reply_token, reply_msg)


@app.post("/")
async def handle_callback(request: Request):
    signature = request.headers["X-Line-Signature"]

    # Get request body as text
    body = await request.body()
    body = body.decode()

    try:
        events = parser.parse(body, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    for event in events:
        # Handle FollowEvent (when user adds the bot)
        if isinstance(event, FollowEvent):
            await handle_follow_event(event)
        # Handle PostbackEvent (e.g., delete file button clicks)
        elif isinstance(event, PostbackEvent):
            await handle_postback(event)
        # Handle MessageEvent
        elif isinstance(event, MessageEvent):
            if event.message.type == "text":
                # Process text message
                await handle_text_message(event, event.message)
            elif event.message.type == "file":
                # Process file message (upload to file search store)
                await handle_document_message(event, event.message)
            elif event.message.type == "image":
                # Process image message (analyze with Gemini vision)
                await handle_image_message(event, event.message)
            else:
                continue
        else:
            continue

    return "OK"


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    await client_session.close()
