from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import os
import requests
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import pytz
from tzlocal import get_localzone

# Gmail/Calendar OAuth2 imports
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import base64
from serpapi import GoogleSearch

load_dotenv()
FANAR_API_KEY = os.getenv("FANAR_API_KEY")
GMAIL_CLIENT_ID = os.getenv("GMAIL_CLIENT_ID")
GMAIL_CLIENT_SECRET = os.getenv("GMAIL_CLIENT_SECRET")
GMAIL_REDIRECT_URI = os.getenv("GMAIL_REDIRECT_URI")
GMAIL_REFRESH_TOKEN = os.getenv("GMAIL_REFRESH_TOKEN")
CALENDAR_ID = "hackathonfanar@gmail.com"
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

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

class ImageGeneratePayload(BaseModel):
    prompt: str

class WebSearchPayload(BaseModel):
    query: str

class GenerateImageAndSendEmailPayload(BaseModel):
    prompt: str
    recipient: str
    subject: str
    body: str

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
def mcp_translate_text(payload: TranslatePayload):
    headers = {
        "Authorization": f"Bearer {FANAR_API_KEY}",
        "Content-Type": "application/json",
    }
    json_data = {
        "model": "Fanar-Shaheen-MT-1",
        "text": payload.text,
        "langpair": payload.langpair,
        "preprocessing": payload.preprocessing,
    }
    try:
        response = requests.post("https://api.fanar.qa/v1/translations", headers=headers, json=json_data)
        response.raise_for_status()
        return {"result": response.json()}
    except requests.exceptions.RequestException as e:
        status_code = response.status_code if 'response' in locals() else 500
        error_message = f"Request Error: {str(e)}"
        if 'response' in locals() and response.text:
            error_message += f" - {response.text}"
        raise HTTPException(status_code=status_code, detail=error_message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API Error: {str(e)}")

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

@app.post("/mcp/format_professional_arabic_email")
def mcp_format_email(payload: FormatEmailPayload):
    """Wraps text in a professional Arabic email format."""
    salutation = "عزيزي"
    if payload.recipient_name:
        salutation = f"عزيزي {payload.recipient_name},"

    closing = "مع خالص التقدير،"
    
    formatted_body = f"""
{salutation}

{payload.body}

{closing}
فريق فنار
"""
    return {"result": "Email body formatted.", "formatted_body": formatted_body.strip()}

@app.post("/mcp/generate_image")
def mcp_generate_image(payload: ImageGeneratePayload):
    """Generates an image from a prompt."""
    try:
        response = fanar_client.images.generate(
            model="Fanar-ImageGen-1",
            prompt=payload.prompt,
            response_format="b64_json"
        )
        image_b64 = response.data[0].b64_json
        if not image_b64:
            raise HTTPException(status_code=500, detail="API returned no image data.")
        return {"result": "Image generated successfully", "image_b64": image_b64}
    except Exception as e:
        # Log the exception for debugging
        print(f"Error in image generation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate image: {str(e)}")

@app.post("/mcp/web_search")
def mcp_web_search(payload: WebSearchPayload):
    """Performs a web search using SerpApi and returns the results."""
    if not SERPAPI_API_KEY:
        raise HTTPException(status_code=500, detail="SERPAPI_API_KEY not set in environment.")
    
    try:
        params = {
            "api_key": SERPAPI_API_KEY,
            "q": payload.query,
            "engine": "google",
            "google_domain": "google.com",
            "gl": "us",
            "hl": "en"
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        
        # Extract and format the organic results
        organic_results = results.get("organic_results", [])
        formatted_results = []
        for result in organic_results[:5]: # Get top 5 results
            formatted_results.append({
                "title": result.get("title"),
                "link": result.get("link"),
                "snippet": result.get("snippet")
            })
            
        return {"result": "Search completed", "results": formatted_results}
        
    except Exception as e:
        print(f"Error in web search: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to perform web search: {str(e)}")

@app.post("/mcp/generate_image_and_send_email")
def mcp_generate_image_and_send_email(payload: GenerateImageAndSendEmailPayload):
    """Generates an image and sends it in an email."""
    try:
        # Step 1: Generate the image by calling the other endpoint's function
        image_payload = ImageGeneratePayload(prompt=payload.prompt)
        image_response = mcp_generate_image(image_payload)
        
        image_b64 = image_response.get("image_b64")
        if not image_b64:
            raise HTTPException(status_code=500, detail="Image generation failed, no image data returned.")

        # Step 2: Send the email with the image
        email_payload = GmailPayload(
            recipient=payload.recipient,
            subject=payload.subject,
            body=payload.body,
            image_b64=image_b64
        )
        email_response = mcp_send_gmail(email_payload)
        return email_response

    except HTTPException as e:
        # Re-raise HTTPException to propagate the error response
        raise e
    except Exception as e:
        logger.error(f"Error in generate_image_and_send_email: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

def generate_image_and_send_email(prompt: str, recipient: str, subject: str, body: str):
    """Generates an image and sends it in an email."""
    try:
        # Step 1: Generate the image
        image_response = mcp_generate_image(ImageGeneratePayload(prompt=prompt))
        if "error" in image_response:
            return image_response  # Propagate the error
        
        image_b64 = image_response.get("image_b64")
        if not image_b64:
            return {"error": "Image generation failed, no image data returned."}

        # Step 2: Send the email with the image
        email_response = mcp_send_gmail(GmailPayload(
            subject=subject,
            body=body,
            recipient=recipient,
            image_b64=image_b64
        ))
        return email_response

    except Exception as e:
        logger.error(f"Error in generate_image_and_send_email: {e}")
        return {"error": str(e)}

def translate_text(text: str, target_lang: str):
    """Translates text to a target language."""
    # ... existing code ...
    return {"result": "Translation completed", "translated_text": translated_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mcp_server:app", host="0.0.0.0", port=8000, reload=True)