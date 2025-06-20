from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import os
import requests
from dotenv import load_dotenv
from openai import OpenAI

# Gmail/Calendar OAuth2 imports
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from email.mime.text import MIMEText
import base64
from serpapi import GoogleSearch

load_dotenv()
FANAR_API_KEY = os.getenv("FANAR_API_KEY")
GMAIL_CLIENT_ID = os.getenv("GMAIL_CLIENT_ID")
GMAIL_CLIENT_SECRET = os.getenv("GMAIL_CLIENT_SECRET")
GMAIL_REDIRECT_URI = os.getenv("GMAIL_REDIRECT_URI")
GMAIL_REFRESH_TOKEN = os.getenv("GMAIL_REFRESH_TOKEN")
CALENDAR_ID = "danijeun@gmail.com"
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

app = FastAPI(title="MCP REST Server (FastAPI)")

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

class CalendarCreatePayload(BaseModel):
    summary: str
    start: str  # ISO format string
    end: str    # ISO format string
    description: Optional[str] = ""
    location: Optional[str] = ""

class CalendarSearchPayload(BaseModel):
    query: Optional[str] = ""
    max_results: Optional[int] = 5
    time_min: Optional[str] = None
    time_max: Optional[str] = None

class CalendarUpdatePayload(BaseModel):
    event_id: str
    updates: Dict[str, Any]

class CalendarDeletePayload(BaseModel):
    event_id: str

class ImageGeneratePayload(BaseModel):
    prompt: str

class WebSearchPayload(BaseModel):
    query: str

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
    to_email = "danijeun@gmail.com"  # Only allow this recipient

    if not all([GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET, GMAIL_REDIRECT_URI, GMAIL_REFRESH_TOKEN]):
        raise HTTPException(status_code=500, detail="Gmail OAuth credentials not set in environment.")

    try:
        creds = Credentials(
            None,
            refresh_token=GMAIL_REFRESH_TOKEN,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=GMAIL_CLIENT_ID,
            client_secret=GMAIL_CLIENT_SECRET,
            scopes=["https://www.googleapis.com/auth/gmail.send"],
        )
        service = build("gmail", "v1", credentials=creds)
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

@app.post("/mcp/calendar_create_event")
def mcp_calendar_create_event(payload: CalendarCreatePayload):
    creds = Credentials(
        None,
        refresh_token=GMAIL_REFRESH_TOKEN,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=GMAIL_CLIENT_ID,
        client_secret=GMAIL_CLIENT_SECRET,
        scopes=["https://www.googleapis.com/auth/calendar"]
    )
    service = build("calendar", "v3", credentials=creds)
    event = {
        "summary": payload.summary,
        "location": payload.location,
        "description": payload.description,
        "start": {"dateTime": payload.start, "timeZone": "UTC"},
        "end": {"dateTime": payload.end, "timeZone": "UTC"},
    }
    try:
        created_event = service.events().insert(calendarId=CALENDAR_ID, body=event).execute()
        return {"result": "Event created", "event": created_event}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create event: {str(e)}")

@app.post("/mcp/calendar_search_event")
def mcp_calendar_search_event(payload: CalendarSearchPayload):
    creds = Credentials(
        None,
        refresh_token=GMAIL_REFRESH_TOKEN,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=GMAIL_CLIENT_ID,
        client_secret=GMAIL_CLIENT_SECRET,
        scopes=["https://www.googleapis.com/auth/calendar"]
    )
    service = build("calendar", "v3", credentials=creds)
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

@app.post("/mcp/calendar_update_event")
def mcp_calendar_update_event(payload: CalendarUpdatePayload):
    creds = Credentials(
        None,
        refresh_token=GMAIL_REFRESH_TOKEN,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=GMAIL_CLIENT_ID,
        client_secret=GMAIL_CLIENT_SECRET,
        scopes=["https://www.googleapis.com/auth/calendar"]
    )
    service = build("calendar", "v3", credentials=creds)
    try:
        event = service.events().get(calendarId=CALENDAR_ID, eventId=payload.event_id).execute()
        event.update(payload.updates)
        updated_event = service.events().update(calendarId=CALENDAR_ID, eventId=payload.event_id, body=event).execute()
        return {"result": "Event updated", "event": updated_event}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update event: {str(e)}")

@app.post("/mcp/calendar_delete_event")
def mcp_calendar_delete_event(payload: CalendarDeletePayload):
    creds = Credentials(
        None,
        refresh_token=GMAIL_REFRESH_TOKEN,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=GMAIL_CLIENT_ID,
        client_secret=GMAIL_CLIENT_SECRET,
        scopes=["https://www.googleapis.com/auth/calendar"]
    )
    service = build("calendar", "v3", credentials=creds)
    try:
        service.events().delete(calendarId=CALENDAR_ID, eventId=payload.event_id).execute()
        return {"result": f"Event {payload.event_id} deleted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete event: {str(e)}")

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mcp_server:app", host="0.0.0.0", port=8000, reload=True)