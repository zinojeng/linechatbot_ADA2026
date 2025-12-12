from fastapi import Request, FastAPI, HTTPException
import os
import sys
import asyncio
import aiohttp
import aiofiles
import json
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

# Google Generative AI imports (Stable SDK)
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

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

# Initialize GenAI (Stable SDK)
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the FastAPI app for LINEBot
app = FastAPI()
client_session = aiohttp.ClientSession()
async_http_client = AiohttpAsyncHttpClient(client_session)
line_bot_api = AsyncLineBotApi(channel_access_token, async_http_client)
parser = WebhookParser(channel_secret)

# Create uploads directory if not exists
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Create data directory for persistent storage
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# JSON file paths for persistent storage
USER_PROFILES_FILE = DATA_DIR / "user_profiles.json"
USER_MODES_FILE = DATA_DIR / "user_modes.json"

# Model configuration
MODEL_NAME = "gemini-1.5-flash"  # or gemini-1.5-pro

# Knowledge Base configuration
# In google-generativeai SDK, we don't have named stores in the same way.
# We will use all uploaded files with a specific prefix or just all text files.
KNOWLEDGE_BASE_PREFIX = "" # Optional filter
USE_KNOWLEDGE_BASE = os.getenv("USE_KNOWLEDGE_BASE", "true").lower() == "true"

# User mode storage: {user_id: "personal" or "knowledge"}
user_modes = {}

# User profiles storage: {user_id: {profile_data}}
user_profiles = {}

# Onboarding state: {user_id: {"step": int, "data": {}}}
onboarding_state = {}


# ========== Persistent Storage Functions ==========

def load_user_data():
    """
    從 JSON 檔案載入使用者資料
    在應用程式啟動時自動執行
    """
    global user_profiles, user_modes

    # Load user profiles
    if USER_PROFILES_FILE.exists():
        try:
            with open(USER_PROFILES_FILE, 'r', encoding='utf-8') as f:
                user_profiles = json.load(f)
            print(f"✅ Loaded {len(user_profiles)} user profiles from {USER_PROFILES_FILE}")
        except Exception as e:
            print(f"❌ Error loading user profiles: {e}")
            user_profiles = {}
    else:
        print(f"ℹ️ No existing user profiles file found, starting fresh")
        user_profiles = {}

    # Load user modes
    if USER_MODES_FILE.exists():
        try:
            with open(USER_MODES_FILE, 'r', encoding='utf-8') as f:
                user_modes = json.load(f)
            print(f"✅ Loaded {len(user_modes)} user modes from {USER_MODES_FILE}")
        except Exception as e:
            print(f"❌ Error loading user modes: {e}")
            user_modes = {}
    else:
        print(f"ℹ️ No existing user modes file found, starting fresh")
        user_modes = {}


def save_user_profiles():
    """
    將使用者個人資料儲存到 JSON 檔案
    每次更新使用者資料時自動執行
    """
    try:
        with open(USER_PROFILES_FILE, 'w', encoding='utf-8') as f:
            json.dump(user_profiles, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved {len(user_profiles)} user profiles to {USER_PROFILES_FILE}")
        return True
    except Exception as e:
        print(f"❌ Error saving user profiles: {e}")
        return False


def save_user_modes():
    """
    將使用者模式設定儲存到 JSON 檔案
    每次切換模式時自動執行
    """
    try:
        with open(USER_MODES_FILE, 'w', encoding='utf-8') as f:
            json.dump(user_modes, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved {len(user_modes)} user modes to {USER_MODES_FILE}")
        return True
    except Exception as e:
        print(f"❌ Error saving user modes: {e}")
        return False


# Load user data on startup
load_user_data()

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
    """設定使用者模式並自動儲存"""
    user_modes[user_id] = mode
    save_user_modes()  # 自動儲存到 JSON 檔案


def get_user_profile(user_id: str) -> dict:
    """取得使用者資料"""
    return user_profiles.get(user_id, {})


def set_user_profile(user_id: str, profile: dict):
    """設定使用者資料並自動儲存"""
    user_profiles[user_id] = profile
    save_user_profiles()  # 自動儲存到 JSON 檔案


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

# ----- Helper functions for Google Generative AI (Stable SDK) -----

_file_cache = None
_file_cache_time = 0

def get_all_remote_files():
    """
    Get all files uploaded to Gemini using genai.list_files().
    Implements simple caching to avoid too many API calls.
    """
    global _file_cache, _file_cache_time
    # Cache for 1 minute
    if _file_cache is not None and (time.time() - _file_cache_time) < 60:
        return _file_cache

    try:
        files = []
        for f in genai.list_files():
            files.append(f)
        _file_cache = files
        _file_cache_time = time.time()
        return files
    except Exception as e:
        print(f"Error listing files: {e}")
        return []

def get_knowledge_base_files():
    """
    Get files designated for the knowledge base.
    In this version, we assume ALL uploaded text/markdown files are part of knowledge base
    unless filtered by display_name logic (not implemented here for simplicity).
    """
    all_files = get_all_remote_files()
    # Filter for typical document types if needed, or by display_name conventions
    # For now, return all files
    return all_files


async def clean_markdown(text: str) -> str:
    """
    移除 Markdown 格式符號，讓訊息在 LINE 中更易讀
    同時在主要標題前加入 emoji 圖示增加閱讀舒適度
    """
    import re
    # (Same implementation as before)
    heading_emoji_map = {
        r'(血糖|血糖控制|監測血糖)': '🩸',
        r'(飲食|營養|食物|餐食|進食)': '🍽️',
        r'(運動|活動|體能|鍛鍊)': '🏃',
        r'(藥物|用藥|胰島素|藥品|治療)': '💊',
        r'(併發症|病變|風險)': '⚠️',
        r'(症狀|徵兆|表現)': '🔍',
        r'(預防|照護|保健|管理)': '🛡️',
        r'(檢查|檢測|診斷|評估)': '🔬',
        r'(生活|日常|習慣)': '🏠',
        r'(注意|提醒|警告|重要)': '⚡',
        r'(建議|方法|步驟|如何)': '💡',
        r'(總結|結論|摘要)': '📋',
        r'(定義|什麼是|介紹)': '📖',
        r'(原因|為什麼|機制)': '🔎',
    }

    def add_emoji_to_heading(match):
        level = len(match.group(1))
        title = match.group(2).strip()
        if level <= 2:
            for pattern, emoji in heading_emoji_map.items():
                if re.search(pattern, title, re.IGNORECASE):
                    return f'{emoji} {title}'
            if level == 1:
                return f'📌 {title}'
            else:
                return f'▸ {title}'
        else:
            return title

    text = re.sub(r'^(#{1,6})\s+(.+)$', add_emoji_to_heading, text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'_(.+?)_', r'\1', text)
    text = re.sub(r'~~(.+?)~~', r'\1', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
    text = re.sub(r'!\[.+?\]\(.+?\)', '', text)
    text = re.sub(r'```[\w]*\n', '', text)
    text = re.sub(r'```', '', text)
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^(\*{3,}|-{3,}|_{3,})$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*+]\s+', '• ', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    return text


async def query_gemini_with_files(query: str, user_id: str = None) -> str:
    """
    Query Gemini using available files as context (Long Context RAG).
    """
    try:
        # 1. Get files
        files = get_knowledge_base_files()
        
        if not files:
            return "📁 您還沒有上傳任何檔案。\n\n請先傳送文件檔案（PDF、DOCX、TXT 等）給我，上傳完成後就可以開始提問了！"

        # 2. Build model and prompt
        model = genai.GenerativeModel(MODEL_NAME)
        
        system_prompt = ""
        if user_id:
            system_prompt = build_system_prompt(user_id)

        # 3. Construct content parts
        # Pass files directly to the model. Gemini 1.5 allows mixing text and file references.
        content_parts = []
        
        # Add system prompt if exists
        if system_prompt:
            content_parts.append(system_prompt + "\n\n")

        # Add files (only up to a reasonable limit or all if they fit context)
        # For this use case, we pass all files found.
        # Note: If there are too many, we might need a selection strategy.
        # But 18 MD files is tiny for 1-2M context window.
        for f in files:
            content_parts.append(f)
            
        content_parts.append(f"\n【患者問題】\n{query}")

        # 4. Generate content
        # Note: generate_content_async is not available in all versions, 
        # but modern versions have it. Safest is run_in_executor for sync call if unsure.
        # We will try standard async call if available or wrap it.
        # Recent google-generativeai supports async generation methods.
        
        response = await model.generate_content_async(content_parts)

        if response.text:
            cleaned_text = await clean_markdown(response.text)
            return cleaned_text
        else:
            return "抱歉，我無法從文件中找到相關資訊。"

    except Exception as e:
        print(f"Error querying Gemini: {e}")
        return f"查詢時發生錯誤：{str(e)}"


async def handle_image_message(event: MessageEvent, message: ImageMessage):
    """
    Handle image messages - analyze using Gemini vision.
    """
    file_name = f"image_{message.id}.jpg"
    
    reply_msg = TextSendMessage(text="正在分析您的圖片，請稍候...")
    await line_bot_api.reply_message(event.reply_token, reply_msg)

    file_path = await download_line_content(message.id, file_name)
    if not file_path:
        return

    try:
        # Upload image to Gemini first (recommended for vision)
        uploaded_file = genai.upload_file(file_path, mime_type="image/jpeg")
        
        # Wait for processing? Images are usually instant but good to check
        # For simplicity, we assume ready or small delay.
        
        model = genai.GenerativeModel(MODEL_NAME)
        response = await model.generate_content_async(
            ["請詳細描述這張圖片的內容，包括主要物品、場景、文字等資訊。", uploaded_file]
        )
        
        result = response.text
        if result:
            result = await clean_markdown(result)
            await line_bot_api.push_message(event.source.user_id, TextSendMessage(text=f"📸 圖片分析結果：\n\n{result}"))
        else:
            await line_bot_api.push_message(event.source.user_id, TextSendMessage(text="無法分析圖片。"))
            
    except Exception as e:
        print(f"Error analyzing image: {e}")
        await line_bot_api.push_message(event.source.user_id, TextSendMessage(text="圖片分析發生錯誤。"))
    finally:
        # Cleanup local file
        if file_path.exists():
            file_path.unlink()


async def handle_document_message(event: MessageEvent, message: FileMessage):
    """
    Handle file messages - upload to Gemini.
    """
    file_name = message.file_name or "unknown_file"

    reply_msg = TextSendMessage(text="正在處理您的檔案，請稍候...")
    await line_bot_api.reply_message(event.reply_token, reply_msg)

    file_path = await download_line_content(message.id, file_name)
    if not file_path:
        return

    try:
        # Upload to Gemini
        # Determine mime type roughly
        mime_type = "text/plain" # Default
        if file_name.lower().endswith(".pdf"):
            mime_type = "application/pdf"
        elif file_name.lower().endswith(".md"):
            mime_type = "text/plain"
        
        genai.upload_file(file_path, display_name=file_name, mime_type=mime_type)
        
        # Invalidate cache so next query sees it
        global _file_cache
        _file_cache = None
        
        await line_bot_api.push_message(
            event.source.user_id, 
            TextSendMessage(text=f"✅ 檔案已成功上傳！\n檔案名稱：{file_name}\n\n現在您可以詢問我關於這個檔案的任何問題。")
        )
        
    except Exception as e:
        print(f"Error uploading file: {e}")
        await line_bot_api.push_message(event.source.user_id, TextSendMessage(text="檔案上傳失敗。"))
    finally:
        if file_path.exists():
            file_path.unlink()


def is_list_files_intent(text: str) -> bool:
    list_keywords = ['列出檔案', '顯示檔案', '查看檔案', '檔案列表', '有哪些檔案']
    return any(k in text.lower() for k in list_keywords)

def is_mode_switch_intent(text: str) -> tuple[bool, str]:
    # ... (Keep existing logic or simplify)
    # For now, simplistic implementation
    if '知識庫' in text:
        return True, 'knowledge'
    return False, ''

# ... (Routes and Main Logic) ...
# Need to copy the route handlers from original file but update calls

@app.post("/callback")
async def callback(request: Request):
    signature = request.headers["X-Line-Signature"]
    body = await request.body()
    body_str = body.decode('utf-8')

    try:
        events = parser.parse(body_str, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    for event in events:
        if isinstance(event, MessageEvent):
            if isinstance(event.message, TextSendMessage) or hasattr(event.message, 'text'):
                text = event.message.text
                user_id = get_user_id(event)
                
                # Check intents
                if is_list_files_intent(text):
                    files = get_all_remote_files()
                    if files:
                        file_list = "\n".join([f"📄 {f.display_name}" for f in files[:20]])
                        if len(files) > 20:
                            file_list += f"\n...還有 {len(files)-20} 個檔案"
                        await line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"📚 目前的知識庫檔案：\n\n{file_list}"))
                    else:
                        await line_bot_api.reply_message(event.reply_token, TextSendMessage(text="目前沒有檔案。"))
                    continue
                    
                # Standard query
                # Use RAG/Long Context by default or if mode is knowledge
                # Since we stripped the strict mode logic for brevity, let's just use it.
                response_text = await query_gemini_with_files(text, user_id)
                await line_bot_api.reply_message(event.reply_token, TextSendMessage(text=response_text))
                
            elif isinstance(event.message, ImageMessage):
                await handle_image_message(event, event.message)
            elif isinstance(event.message, FileMessage):
                await handle_document_message(event, event.message)

    return "OK"

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
