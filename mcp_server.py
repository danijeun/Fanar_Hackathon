from fastapi import FastAPI, HTTPException, Request, Body
from pydantic import BaseModel
from typing import Dict, Any, Optional
import os
import requests
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import pytz
from tzlocal import get_localzone
import json
from telegram_bot import send_telegram_notification
import asyncio
import base64
import serpapi
import threading
import time
import logging
import re

# Gmail/Calendar OAuth2 imports
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from google.cloud import pubsub_v1
from google.api_core.exceptions import AlreadyExists
from concurrent.futures import TimeoutError

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
FANAR_API_KEY = os.getenv("FANAR_API_KEY")
GMAIL_CLIENT_ID = os.getenv("GMAIL_CLIENT_ID")
GMAIL_CLIENT_SECRET = os.getenv("GMAIL_CLIENT_SECRET")
GMAIL_REDIRECT_URI = os.getenv("GMAIL_REDIRECT_URI")
GMAIL_REFRESH_TOKEN = os.getenv("GMAIL_REFRESH_TOKEN")
CALENDAR_ID = "hackathonfanar@gmail.com"
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

app = FastAPI(title="MCP REST Server (FastAPI)")

# --- Google API Utility ---
def _get_google_service(api_name: str, api_version: str, scopes: list[str]):
    """Helper function to create a Google API service client."""
    if not GMAIL_REFRESH_TOKEN:
        raise HTTPException(status_code=500, detail="GMAIL_REFRESH_TOKEN not set in environment.")
    
    try:
        creds = Credentials(
            None,
            refresh_token=GMAIL_REFRESH_TOKEN,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=GMAIL_CLIENT_ID,
            client_secret=GMAIL_CLIENT_SECRET,
            scopes=scopes,
        )
        service = build(api_name, api_version, credentials=creds)
        return service
    except Exception as e:
        # This could be due to invalid credentials, expired refresh token, etc.
        print(f"[ERROR] Failed to create Google service '{api_name} v{api_version}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to authenticate with Google: {e}")

# Initialize Fanar OpenAI Client
fanar_client = OpenAI(
    base_url="https://api.fanar.qa/v1",
    api_key=FANAR_API_KEY,
)

class TranslatePayload(BaseModel):
    text: str
    langpair: str
    preprocessing: str = "default"

class GmailPayload(BaseModel):
    subject: str
    body: str
    recipient: str
    image_b64: Optional[str] = None

class CreateCalendarEventPayload(BaseModel):
    summary: str
    start: str
    end: str
    description: Optional[str] = None
    location: Optional[str] = None

class ListCalendarEventsPayload(BaseModel):
    query: Optional[str] = None
    max_results: int = 10
    time_min: Optional[str] = None
    time_max: Optional[str] = None

class FormatEmailPayload(BaseModel):
    body: str
    recipient_name: Optional[str] = None

class FormatEnglishEmailPayload(BaseModel):
    body: str
    recipient_name: str

class FormatArabicEmailPayload(BaseModel):
    body: str
    recipient_name: str

class EmailAgentPayload(BaseModel):
    body: str
    recipient_name: str
    recipient_email: str

def to_utc_iso(local_dt_str: str, local_fmt: str = "%Y-%m-%d %H:%M") -> str:
    """Converts a local datetime string to a UTC ISO formatted string for Google Calendar."""
    try:
        local_dt = datetime.strptime(local_dt_str, local_fmt)
        local_tz = get_localzone()
        local_aware = local_tz.localize(local_dt) if hasattr(local_tz, 'localize') else local_dt.replace(tzinfo=local_tz)
        utc_dt = local_aware.astimezone(pytz.UTC)
        return utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, TypeError) as e:
        print(f"[ERROR] Could not parse date string '{local_dt_str}'. Error: {e}")
        # Re-raise as HTTPException so the client gets a clean error
        raise HTTPException(status_code=400, detail=f"Invalid date format for '{local_dt_str}'. Please use 'YYYY-MM-DD HH:MM'.")

@app.post("/mcp/translate_text")
def mcp_translate_text(payload: dict):
    """Translates text to a target language using the Fanar LLM."""
    text = payload.get("text")
    # Accept 'target_lang' (preferred) or fallback to 'langpair'
    target_lang = payload.get("target_lang") or payload.get("langpair")
    if not text or not target_lang:
        raise HTTPException(status_code=422, detail="Both 'text' and 'target_lang' are required.")
    prompt = f"Translate the following text to {target_lang}:\n\n{text}"
    try:
        print(f"[DEBUG] Translate prompt: {prompt}")
        response = fanar_client.chat.completions.create(
            model="Fanar-S-1-7B",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.3
        )
        translated = response.choices[0].message.content
        return {"result": "Text translated successfully", "translated": translated}
    except Exception as e:
        print(f"[DEBUG] Exception in /mcp/translate_text: {e}")
        import traceback
        traceback.print_exc()
        return {"result": "[DEBUG] Exception occurred", "translated": f"[DEBUG] {str(e)}"}

@app.post("/mcp/send_gmail")
def mcp_send_gmail(payload: GmailPayload):
    to_email = payload.recipient

    if not all([GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET, GMAIL_REDIRECT_URI, GMAIL_REFRESH_TOKEN]):
        raise HTTPException(status_code=500, detail="Gmail OAuth credentials not set in environment.")

    try:
        service = _get_google_service("gmail", "v1", ["https://www.googleapis.com/auth/gmail.send"])
        
        if payload.image_b64:
            message = MIMEMultipart()
            msg_text = MIMEText(payload.body)
            message.attach(msg_text)
            
            # Decode the base64 string
            image_data = base64.b64decode(payload.image_b64)
            image = MIMEImage(image_data, name="image.png")
            message.attach(image)
        else:
            message = MIMEText(payload.body)

        message["to"] = to_email
        message["from"] = "me"
        message["subject"] = payload.subject
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        
        send_result = service.users().messages().send(
            userId="me",
            body={"raw": raw_message}
        ).execute()
        return {"result": f"Email sent to {to_email} with subject '{payload.subject}'.", "gmail_id": send_result.get('id')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")

@app.post("/mcp/create_calendar_event")
def mcp_create_calendar_event(payload: CreateCalendarEventPayload):
    service = _get_google_service("calendar", "v3", ["https://www.googleapis.com/auth/calendar"])

    # Convert start and end times to the required format
    try:
        start_utc = to_utc_iso(payload.start)
        end_utc = to_utc_iso(payload.end)
    except HTTPException as e:
        # Forward the specific date parsing error to the client
        raise e

    event = {
        "summary": payload.summary,
        "location": payload.location,
        "description": payload.description,
        "start": {"dateTime": start_utc, "timeZone": "UTC"},
        "end": {"dateTime": end_utc, "timeZone": "UTC"},
    }
    try:
        created_event = service.events().insert(calendarId=CALENDAR_ID, body=event).execute()
        return {"result": "Event created", "event": created_event}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create event: {str(e)}")

@app.post("/mcp/list_calendar_events")
def mcp_list_calendar_events(payload: ListCalendarEventsPayload):
    service = _get_google_service("calendar", "v3", ["https://www.googleapis.com/auth/calendar"])
    try:
        list_params = {
            "calendarId": CALENDAR_ID,
            "q": payload.query,
            "maxResults": payload.max_results,
            "singleEvents": True,
            "orderBy": "startTime"
        }
        
        if payload.time_min:
            list_params["timeMin"] = payload.time_min
        if payload.time_max:
            list_params["timeMax"] = payload.time_max
            
        events_result = service.events().list(**list_params).execute()
        events = events_result.get("items", [])
        return {"result": "Events found", "events": events}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search events: {str(e)}")


def check_new_emails():
    """Periodically checks for new, unread emails and sends notifications."""
    print("--- Starting Gmail polling thread ---")
    
    while True:
        try:
            if not TELEGRAM_CHAT_ID:
                print("TELEGRAM_CHAT_ID not set, polling is disabled.")
                time.sleep(60)
                continue

            print("... Checking for new emails ...")
            
            # Use .modify scope to be able to mark emails as read
            service = _get_google_service("gmail", "v1", ["https://www.googleapis.com/auth/gmail.modify"])
            
            # List all unread messages in the inbox
            results = service.users().messages().list(userId='me', q='is:unread in:inbox').execute()
            messages = results.get('messages', [])

            if not messages:
                print("... No new emails found.")
            else:
                print(f"--- Found {len(messages)} new email(s)! ---")
                for msg_summary in messages:
                    msg_id = msg_summary['id']
                    
                    # Fetch full message details
                    full_message = service.users().messages().get(userId='me', id=msg_id).execute()
                    headers = full_message['payload']['headers']
                    subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), 'No Subject')
                    sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), 'Unknown Sender')
                    # Extract the plain text body (if available)
                    body = ''
                    payload = full_message.get('payload', {})
                    if 'parts' in payload:
                        for part in payload['parts']:
                            if part.get('mimeType') == 'text/plain':
                                body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
                                break
                    elif payload.get('mimeType') == 'text/plain':
                        body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', errors='ignore')
                    notification_text = f"📬 New Email!\n\n*From:* {sender}\n*Subject:* {subject}"
                    # Send notification with summarize button
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(send_telegram_notification(TELEGRAM_CHAT_ID, notification_text, email_body=body, subject=subject))
                    loop.close()
                    print(f"Sent notification for message ID: {msg_id}")
                    # Mark the message as read by removing the 'UNREAD' label
                    service.users().messages().modify(userId='me', id=msg_id, body={'removeLabelIds': ['UNREAD']}).execute()

            # Wait for 30 seconds before checking again
            time.sleep(10)
            
        except Exception as e:
            print(f"An error occurred in the email polling thread: {e}")
            # Wait longer before retrying if an error occurs
            time.sleep(60)


@app.on_event("startup")
def startup_event():
    """On app startup, run the email poller in a separate thread."""
    polling_thread = threading.Thread(target=check_new_emails, daemon=True)
    polling_thread.start()

@app.post("/mcp/arabic_email_agent")
def mcp_arabic_email_agent(payload: EmailAgentPayload):
    # Self-send detection
    user_body = payload.body or ""
    if re.search(r"to myself|to me|to my email|send me an email|ارسل لي|ارسل الى بريدي", user_body, re.IGNORECASE):
        payload.recipient_email = "hackathonfanar@gmail.com"
    # Generic salutation if recipient_name is missing
    recipient_name = payload.recipient_name.strip() if payload.recipient_name else None
    if not recipient_name:
        salutation = "عزيزي،"
        salutation_instruction = "إذا لم يتم توفير اسم المستلم من قبل المستخدم بشكل صريح، استخدم فقط تحية عامة 'عزيزي،' ولا تحاول أبداً تخمين أو استنتاج اسم المستلم من البريد الإلكتروني أو السياق أو أي مصدر آخر. لا تكتب أي اسم أو نص بين قوسين أو أقواس زاوية أو مربعة أو أي صيغة مثل <اسم المستلم> أو [اسم المستلم]. لا تضع أي عنصر نائب أو نص توضيحي."
    else:
        salutation = f"عزيزي {recipient_name},"
        salutation_instruction = ""
    negative_instruction = "ممنوع تمامًا إضافة أي شروحات أو اعتذارات أو ذكر أنك نموذج لغوي أو أي عبارات توضيحية أو تبريرية أو محتوى مخترع. فقط أعد صياغة ونسق رسالة المستخدم الرئيسية كبريد إلكتروني احترافي، وانقل نية المستخدم الأصلية فقط، ولا شيء أكثر."
    subject_instruction = "يجب أن تبدأ كل رسالة بريد إلكتروني بسطر موضوع واضح في الأعلى، مكتوب هكذا: 'موضوع: ...'، ولا يجوز ترك الموضوع فارغًا أبدًا."
    prompt = f"""
مهمتك هي كتابة بريد إلكتروني احترافي كامل باللغة العربية بناءً بدقة على طلب المستخدم أدناه.

**التعليمات:**
- أنشئ البريد الإلكتروني بالكامل، بما في ذلك سطر الموضوع، التحية، النص الأساسي، والخاتمة.
- يجب أن يكون البريد الإلكتروني جاهزًا للإرسال فورًا.
- لا تدرج أي تعليقات أو ملاحظات تعليمية أو نصوص بين قوسين أو أقواس زاوية أو مربعة أو عناصر نائبة.
- استخدم نبرة رسمية وواضحة.
- التحية: ابدأ بـ '{salutation}'
- الخاتمة: أنهِ البريد الإلكتروني بـ 'مع خالص التقدير،' متبوعة بـ 'فَنار'.
- لا تجب أو تشرح أو تضف اقتراحات على طلب المستخدم. لا تخترع أو تقترح حلولاً. فقط أعد صياغة ونسق رسالة المستخدم الرئيسية كبريد إلكتروني احترافي. يجب أن ينقل الناتج نية المستخدم الأصلية فقط، ولا شيء أكثر.
{salutation_instruction}
{negative_instruction}
{subject_instruction}

**طلب المستخدم:**
"{payload.body}"

الآن، اكتب البريد الإلكتروني الكامل والنهائي باللغة العربية، جاهز للإرسال. لا تضف أي شيء غير موجود في رسالة المستخدم.
"""
    response = fanar_client.chat.completions.create(
        model="Fanar-C-1-8.7B",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0.7
    )
    generated_email = response.choices[0].message.content
    lines = generated_email.strip().split('\n')
    subject = "بدون موضوع"
    body = generated_email
    if lines and lines[0].startswith("موضوع:"):
        subject = lines[0].split(":", 1)[1].strip()
        body = "\n".join(lines[1:]).strip()
    else:
        # Prepend default subject if missing
        body = f"موضوع: {subject}\n" + body
    # Do NOT send the email, just return the draft
    return {"email": generated_email, "subject": subject, "body": body}

@app.post("/mcp/english_email_agent")
def mcp_english_email_agent(payload: EmailAgentPayload):
    # Self-send detection
    user_body = payload.body or ""
    if re.search(r"to myself|to me|to my email|send me an email|ارسل لي|ارسل الى بريدي", user_body, re.IGNORECASE):
        payload.recipient_email = "hackathonfanar@gmail.com"
    # Generic salutation if recipient_name is missing
    recipient_name = payload.recipient_name.strip() if payload.recipient_name else None
    if not recipient_name:
        salutation = "Dear,"
        salutation_instruction = "If the user does NOT explicitly provide a recipient name, use ONLY a generic salutation 'Dear,'. NEVER guess, invent, or infer a name from the email address or context. Do not write any name, placeholder, or text in brackets, angle brackets, or like <Recipient's Name> or [Recipient's Name]. Do not include any placeholder or instructional text."
    else:
        salutation = f"Dear {recipient_name},"
        salutation_instruction = ""
    negative_instruction = "You must NEVER add explanations, disclaimers, or say you are a language model. Do NOT invent, elaborate, or add any content not present in the user's request. Never guess or infer a name. Only restate, rephrase, and format the user's main message as a professional email. The output must only convey the user's original intent, nothing more."
    subject_instruction = "Every email must start with a clear subject line at the very top, formatted as: 'Subject: ...'. The subject must never be omitted or left blank."
    prompt = f"""
You are a professional email assistant. Your task is to write a complete, professional English email based strictly on the user's request below.

**Instructions:**
- Generate the entire email, including a subject line, salutation, body, and closing.
- The email must be ready to send immediately.
- Do not include any instructional comments, parentheses, brackets, angle brackets, or placeholders.
- Use a professional and clear tone.
- Salutation: Start with '{salutation}'
- Closing: End with 'Best regards,' followed by 'Sa'i'.
- Do NOT answer, elaborate, or add suggestions to the user's query. Do NOT invent or propose solutions. Only restate, rephrase, and format the user's main message as a professional email. The output must only convey the user's original intent, nothing more.
{salutation_instruction}
{negative_instruction}
{subject_instruction}

**User's Request:**
"{payload.body}"

Now, write the complete and final email in English, ready to send. Do not add anything that is not in the user's message.
"""
    response = fanar_client.chat.completions.create(
        model="Fanar-C-1-8.7B",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0.7
    )
    generated_email = response.choices[0].message.content
    lines = generated_email.strip().split('\n')
    subject = "No Subject"
    body = generated_email
    if lines and lines[0].lower().startswith("subject:"):
        subject = lines[0].split(":", 1)[1].strip()
        body = "\n".join(lines[1:]).strip()
    else:
        # Prepend default subject if missing
        body = f"Subject: {subject}\n" + body
    # Do NOT send the email, just return the draft
    return {"email": generated_email, "subject": subject, "body": body}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mcp_server:app", host="0.0.0.0", port=8000, reload=True)