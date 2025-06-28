import os
import io
import asyncio
import re
from dotenv import load_dotenv
import logging
import requests
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    ConversationHandler,
    CallbackQueryHandler,
)
from telegram.request import HTTPXRequest
from backend.app import (
    get_agent_response,
    get_vision_response,
    generate_image_from_prompt,
    summarize_conversation_history,
    summarize_messages_with_llm,
    extract_event_from_text,
)
from backend.database import (
    init_db,
    get_conversation_history,
    update_conversation_history,
    clear_conversation_history,
    save_email_body,
    get_email_body,
    save_short_id_mapping,
    get_short_id_mapping,
    delete_short_id_mapping,
    save_schedule_event_mapping,
    get_schedule_event_mapping,
    delete_schedule_event_mapping,
)
import nest_asyncio
from openai import OpenAI
from pydub import AudioSegment
import uuid
import json
from datetime import datetime, timedelta
from backend.utils import resolve_natural_date

# Apply nest_asyncio to allow asyncio to be nested, which is needed by the telegram library
nest_asyncio.apply()

# --- Setup ---
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
FANAR_API_KEY = os.getenv("FANAR_API_KEY")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000")

# --- Logging Setup ---
# Set the logging level for all loggers to WARNING to suppress INFO and DEBUG messages
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.WARNING
)
# Keep httpx logging at WARNING as it's particularly verbose
logging.getLogger("httpx").setLevel(logging.WARNING)

# Get the root logger and set its level to WARNING.
# This is a more forceful way to ensure no DEBUG/INFO logs from libraries get through.
logging.getLogger().setLevel(logging.WARNING)

# Define states for conversation
(
    PROMPT,
    CHECK_PLACEHOLDERS,
    FILL_PLACEHOLDER,
    CONFIRM_SEND,
    AWAITING_RECIPIENT,
) = range(5)

logger = logging.getLogger(__name__)

# --- Fanar TTS ---
def get_tts_audio(text):
    """Generates TTS audio from text using the Fanar API."""
    try:
        client = OpenAI(
            base_url="https://api.fanar.qa/v1",
            api_key=FANAR_API_KEY,
        )
        response = client.audio.speech.create(
            model="Fanar-Aura-TTS-1",
            input=text,
            voice="default",
        )
        # It's better to work with the audio data in memory
        return response.read()
    except Exception as e:
        logger.error(f"Error in Fanar TTS API: {e}")
        return None

# --- Fanar STT ---
def transcribe_audio(audio_bytes):
    """Transcribes audio using the Fanar API."""
    try:
        client = OpenAI(
            base_url="https://api.fanar.qa/v1",
            api_key=FANAR_API_KEY,
        )
        # The API expects a file-like object, so we use io.BytesIO
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "voice_message.ogg" # The API needs a filename
        transcript = client.audio.transcriptions.create(
            model="Fanar-Aura-STT-1", 
            file=audio_file,
        )
        return transcript.text
    except Exception as e:
        logger.error(f"Error in Fanar STT API: {e}")
        return None

# Temporary in-memory store for email drafts (keyed by chat_id)
email_draft_store = {}

# --- Bot Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message and asks the user to log in."""
    user = update.effective_user
    chat_id = update.message.chat_id
    clear_conversation_history(chat_id)  # Reset history on start

    await update.message.reply_html(
        rf"Hi {user.mention_html()}! I'm your professional AI email assistant, powered by Fanar. "
        "I can help you compose, format, and send emails to anyone from hackathonfanar@gmail.com. "
        "To get started, just tell me what email you'd like to send, or use the /email command."
    )

async def email_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Starts the email creation conversation."""
    await update.message.reply_text("Please describe the email you want me to write.")
    return PROMPT

async def handle_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Gets the user's email prompt, generates the email, and initiates the placeholder check."""
    prompt = update.message.text
    await update.message.reply_text("👍 Got it. Generating a draft now...")

    # Try to extract a recipient's name from the initial prompt
    recipient_name = None
    name_match = re.search(r"to ([\w\s]+?)(?=(with the subject|about|that says))", prompt, re.IGNORECASE)
    if name_match:
        recipient_name = name_match.group(1).strip()
        # Capitalize the name properly
        recipient_name = ' '.join([name.capitalize() for name in recipient_name.split()])
        
    try:
        # Use an HTTP request to call the generate_email endpoint
        payload = {"body": prompt}
        if recipient_name:
            payload["recipient_name"] = recipient_name

        response = requests.post(
            f"{MCP_SERVER_URL}/mcp/generate_email",
            json=payload
        )
        response.raise_for_status()  # Raise an exception for bad status codes
        email_data = response.json()
        
        email_body = email_data.get("email")
        if not email_body:
            await update.message.reply_text("Sorry, I couldn't generate an email from that. Please try again with a different description.")
            return ConversationHandler.END

        context.user_data["email_body"] = email_body
        
        # Move to the placeholder checking state
        return await check_for_placeholders(update, context)

    except Exception as e:
        logger.error(f"Error generating email: {e}")
        await update.message.reply_text("I encountered an error trying to generate the email. Please try again later.")
        return ConversationHandler.END

async def check_for_placeholders(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Checks the email for placeholders and asks the user to fill them."""
    email_body = context.user_data.get("email_body", "")
    
    # Simple regex to find anything in square brackets
    placeholders = re.findall(r"\[(.*?)\]", email_body)
    
    if placeholders:
        # For simplicity, just ask for the first one found
        placeholder_key = placeholders[0]
        context.user_data["current_placeholder"] = placeholder_key
        
        # Make the question more user-friendly
        question = f"What should I put for '{placeholder_key}'?"
        if "recipient" in placeholder_key.lower() or "name" in placeholder_key.lower():
            question = f"Who is this email for? (What is the recipient's name?)"
        elif "subject" in placeholder_key.lower():
            question = "What should the subject of the email be?"

        await update.message.reply_text(question)
        return FILL_PLACEHOLDER
    else:
        # No placeholders, move to confirmation
        # Use code formatting for the email body
        await update.message.reply_text(f"```\n{email_body}\n```", parse_mode='MarkdownV2')
        await update.message.reply_text("Should I send this? (yes/no)")
        return CONFIRM_SEND

async def handle_placeholder_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Fills the current placeholder with the user's input."""
    user_input = update.message.text
    placeholder = context.user_data.get("current_placeholder")
    
    if placeholder:
        # Replace the first occurrence of the placeholder
        email_body = context.user_data.get("email_body", "")
        context.user_data["email_body"] = email_body.replace(f"[{placeholder}]", user_input, 1)
    
    # Check for more placeholders
    return await check_for_placeholders(update, context)

async def handle_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles the user's confirmation to send the email."""
    user_input = update.message.text.lower()
    
    if "yes" in user_input:
        await update.message.reply_text("Great! Who should I send this email to? Please provide their email address.")
        return AWAITING_RECIPIENT
    elif "no" in user_input:
        await update.message.reply_text("Okay, I won't send the email. You can start over by using the /email command.")
        context.user_data.clear()
        return ConversationHandler.END
    else:
        await update.message.reply_text("Please answer with 'yes' or 'no'.")
        return CONFIRM_SEND

def extract_subject_and_body(full_email: str) -> tuple[str, str]:
    """Extracts the subject and body from a full email string."""
    # Handle emails that might start with the subject on the first line
    lines = full_email.strip().split('\n')
    first_line = lines[0]
    
    subject = "No Subject"
    body = full_email

    if first_line.lower().startswith("subject:"):
        subject = first_line.split(":", 1)[1].strip()
        body = "\n".join(lines[1:]).strip()
    
    return subject, body

async def handle_recipient(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Gets the recipient's email, sends the email, and ends the conversation."""
    recipient_email = update.message.text
    email_body = context.user_data.get("email_body", "")

    # Basic email validation
    if not re.match(r"[^@]+@[^@]+\.[^@]+", recipient_email):
        await update.message.reply_text("That doesn't look like a valid email address. Please provide a correct email address.")
        return CONFIRM_SEND

    await update.message.reply_text(f"Alright, sending the email to {recipient_email}...")

    subject, body = extract_subject_and_body(email_body)

    try:
        # Use an HTTP request to call the send_gmail endpoint
        payload = {
            "recipient": recipient_email,
            "subject": subject,
            "body": body,
        }
        response = requests.post(
            f"{MCP_SERVER_URL}/mcp/send_gmail",
            json=payload
        )
        response.raise_for_status()
        response_data = response.json()

        if "error" in response_data:
            await update.message.reply_text(f"Sorry, there was an error sending the email: {response_data['error']}")
        else:
            await update.message.reply_text("✅ Email sent successfully!")

    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP Error sending email: {e}")
        await update.message.reply_text(f"I couldn't connect to the backend server to send the email. Error: {e}")
    except Exception as e:
        logger.error(f"Error sending email via MCP: {e}")
        await update.message.reply_text("I encountered a critical error trying to send the email.")
    
    context.user_data.clear()
    return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels and ends the conversation."""
    await update.message.reply_text("Okay, I've cancelled the email.")
    context.user_data.clear()
    return ConversationHandler.END

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles all incoming text and photo messages."""
    user_id = update.message.from_user.id
    chat_id = update.message.chat_id
    user_input = update.message.text
    
    # Get user's conversation history from the database
    history = get_conversation_history(chat_id)
    # Only keep the last 5 messages for context
    history = history[-5:]
    
    # Send a "typing..." notification
    await context.bot.send_chat_action(chat_id=chat_id, action='typing')
    
    # Get the agent's response
    # Since get_agent_response is not an async function, we run it in a thread
    loop = asyncio.get_event_loop()
    text_response, media_response, updated_history = await loop.run_in_executor(
        None, get_agent_response, user_input, history
    )

    # Summarize before saving
    summarized_history = summarize_conversation_history(updated_history)
    update_conversation_history(chat_id, summarized_history)
    
    # --- EMAIL DRAFT APPROVAL WORKFLOW ---
    # Restore the original inline logic for email draft approval
    try:
        # Try to parse a JSON object from the raw response (for tool call results)
        if isinstance(text_response, str) and ("english_email_agent" in text_response or "arabic_email_agent" in text_response or 'email' in text_response):
            # Try to extract the email draft and payload from the tool result
            # Look for a JSON object in the response
            email_payload = None
            email_draft = None
            # Try to find a JSON object in the response
            match = re.search(r'\{\s*"email"\s*:\s*"([\s\S]+?)",\s*"send_result"\s*:\s*\{[\s\S]+?\}\s*\}', text_response)
            if match:
                # This is a fallback, but ideally we want to use the tool result directly
                email_draft = match.group(1)
            # Try to find a tool result in updated_history
            for msg in reversed(updated_history):
                if isinstance(msg, dict) and msg.get('role') == 'system' and 'TOOL_RESULTS' in msg.get('content', ''):
                    try:
                        tool_results = json.loads(msg['content'].split('TOOL_RESULTS:')[-1])
                        for res in tool_results:
                            if res.get('tool') in ('english_email_agent', 'arabic_email_agent') and 'result' in res:
                                # Get the original tool call payload (from res['payload'])
                                tool_payload = res.get('payload', {})
                                # Get the generated email
                                email_draft = res['result'].get('email')
                                # Try to extract subject and body from the draft
                                subject = "No Subject"
                                body = email_draft
                                lines = email_draft.strip().split('\n')
                                if lines and (lines[0].lower().startswith("subject:") or lines[0].startswith("موضوع:")):
                                    subject = lines[0].split(":", 1)[1].strip()
                                    body = "\n".join(lines[1:]).strip()
                                # Build the payload for send_gmail
                                recipient = tool_payload.get("recipient_email")
                                if not recipient:
                                    recipient = extract_email_from_user_input(user_input) or "hackathonfanar@gmail.com"
                                email_payload = {
                                    "recipient": recipient,
                                    "subject": subject,
                                    "body": body
                                }
                                # Store for callback
                                email_draft_store[chat_id] = {
                                    'draft': email_draft,
                                    'recipient': recipient,
                                    'subject': subject,
                                    'body': body
                                }
                                break
                    except Exception:
                        pass
            # If we have a draft, show it to the user with a Send button
            if email_draft:
                # Store the draft and info for later sending (use chat_id as key)
                if chat_id not in email_draft_store:
                    recipient = tool_payload.get("recipient_email")
                    if not recipient:
                        recipient = extract_email_from_user_input(user_input) or "hackathonfanar@gmail.com"
                    email_draft_store[chat_id] = {
                        'draft': email_draft,
                        'recipient': recipient,
                        'subject': subject,
                        'body': body
                    }
                button = InlineKeyboardButton("Send", callback_data=f"send_email:{chat_id}")
                reply_markup = InlineKeyboardMarkup([[button]])
                await update.message.reply_text(f"Here is your email draft:\n\n{email_draft}\n\nPress 'Send' to send this email.", reply_markup=reply_markup)
                return
    except Exception as e:
        logger.error(f"Error in email draft approval workflow: {e}")

    # Send the response back to the user
    if media_response:
        # If there's an image, send it with the text as a caption
        await update.message.reply_photo(photo=media_response, caption=text_response)
    elif text_response:
        # Always show the backend/LLM response, even if it contains an error
        await update.message.reply_text(text_response)
    else:
        # Only show a generic error if there is truly no information
        await update.message.reply_text("Sorry, I couldn't process your request and there is no further information.")

async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles incoming voice messages, transcribes them, and gets a response."""
    chat_id = update.message.chat_id
    
    await context.bot.send_chat_action(chat_id=chat_id, action='typing')
    
    try:
        # Download the voice message
        voice_file = await context.bot.get_file(update.message.voice.file_id)
        voice_bytes = await voice_file.download_as_bytearray()
        
        # Convert the downloaded audio (likely in OGG format) to MP3
        audio = AudioSegment.from_file(io.BytesIO(voice_bytes))
        mp3_bytes = io.BytesIO()
        audio.export(mp3_bytes, format="mp3")
        mp3_bytes.seek(0)
        
        # Transcribe the audio to text
        transcribed_text = transcribe_audio(mp3_bytes.read())
        
        if transcribed_text:
            logger.info(f"Transcribed text: {transcribed_text}")
            await update.message.reply_text(f"Heard: \"{transcribed_text}\"")
            
            # Now, process this text with the agent like a normal message
            history = get_conversation_history(chat_id)
            # Only keep the last 5 messages for context
            history = history[-5:]
            loop = asyncio.get_event_loop()
            text_response, media_response, updated_history = await loop.run_in_executor(
                None, get_agent_response, transcribed_text, history
            )
            # Summarize before saving
            summarized_history = summarize_conversation_history(updated_history)
            update_conversation_history(chat_id, summarized_history)
            
            # Handle the response (could be text or an image)
            if media_response:
                # If there's an image, send it with the text as a caption
                await update.message.reply_photo(photo=media_response, caption=text_response)
            elif text_response:
                # Always show the backend/LLM response, even if it contains an error
                audio_data = get_tts_audio(text_response)
                if audio_data:
                    await update.message.reply_audio(audio=audio_data, title="Fanar Response")
                else:
                    await update.message.reply_text(text_response)
            else:
                await update.message.reply_text("Sorry, I couldn't process your request and there is no further information.")
        else:
            await update.message.reply_text("I'm sorry, I couldn't understand the audio. Please try again.")
            
    except Exception as e:
        logger.error(f"Error processing voice message: {e}")
        await update.message.reply_text("I'm sorry, I encountered an error processing your voice message.")

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles incoming image messages and gets a description from the vision model."""
    chat_id = update.message.chat_id
    
    await context.bot.send_chat_action(chat_id=chat_id, action='typing')
    
    try:
        # Get the highest resolution photo
        photo_file = await context.bot.get_file(update.message.photo[-1].file_id)
        photo_bytes = await photo_file.download_as_bytearray()
        
        # Get the caption as the prompt
        prompt = update.message.caption or "What's in this image? Provide a detailed description."
        
        # Get the vision response
        loop = asyncio.get_event_loop()
        vision_response = await loop.run_in_executor(
            None, get_vision_response, prompt, bytes(photo_bytes)
        )
        
        # Send the response back to the user
        await update.message.reply_text(vision_response)
        
    except Exception as e:
        logger.error(f"Error processing image message: {e}")
        await update.message.reply_text("I'm sorry, I encountered an error processing your image.")

async def generate_image_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /generate command to create an image from a prompt."""
    chat_id = update.message.chat_id
    
    # Extract prompt from the command
    prompt = " ".join(context.args)
    
    if not prompt:
        await update.message.reply_text("Please provide a prompt after the /generate command. \n\nExample: `/generate a cat in a hat`")
        return
        
    await context.bot.send_chat_action(chat_id=chat_id, action='upload_photo')
    await update.message.reply_text(f"🎨 Generating image for prompt: \"{prompt}\"...")
    
    try:
        # Get the image bytes from the backend
        loop = asyncio.get_event_loop()
        image_bytes = await loop.run_in_executor(
            None, generate_image_from_prompt, prompt
        )
        
        if image_bytes:
            # Send the photo
            await update.message.reply_photo(photo=image_bytes)
        else:
            await update.message.reply_text("Sorry, I couldn't generate the image. There might have been an issue with the API.")
            
    except Exception as e:
        logger.error(f"Error processing /generate command: {e}")
        await update.message.reply_text("I'm sorry, I encountered an error while generating your image.")

def get_short_id():
    import uuid
    return str(uuid.uuid4())[:8]

async def send_telegram_notification(chat_id: str, message: str, email_body: str = None, subject: str = None) -> None:
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not set for notification.")
        return
    try:
        application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        reply_markup = None
        short_id = None
        if email_body:
            short_id = get_short_id()
            message_id = subject or str(uuid.uuid4())
            save_email_body(int(chat_id), message_id, email_body)
            save_short_id_mapping(short_id, message_id)
            button = InlineKeyboardButton("Summarize Email", callback_data=f"summarize_email:{short_id}")
            reply_markup = InlineKeyboardMarkup([[button]])
        await application.bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown', reply_markup=reply_markup)
        logger.info(f"Notification sent to {chat_id}")
    except Exception as e:
        logger.error(f"Failed to send Telegram notification to {chat_id}: {e}")

async def summarize_email_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    chat_id = query.message.chat_id
    await query.answer()
    data = query.data
    short_id = None
    if data and data.startswith("summarize_email:"):
        short_id = data.split(":", 1)[1]
    if not short_id or not get_short_id_mapping(short_id):
        await query.edit_message_text("Sorry, I couldn't find the email body to summarize. Please try again with a new notification.")
        return
    message_id = get_short_id_mapping(short_id)
    email_body = get_email_body(int(chat_id), message_id)
    if not email_body:
        await query.edit_message_text("Sorry, I couldn't find the email body to summarize.")
        return
    summary = None
    try:
        summary = summarize_messages_with_llm([
            {"role": "user", "content": email_body}
        ])
    except Exception as e:
        logger.error(f"Error summarizing email: {e}")
        summary = None
    # Try to extract event
    event = None
    try:
        event = extract_event_from_text(email_body)
        logger.warning(f"[DEBUG] Event extraction result: {event}")
    except Exception as e:
        logger.error(f"Error extracting event: {e}")
        event = None
    if summary and event and event.get('title') and event.get('start'):
        import uuid, json
        schedule_id = str(uuid.uuid4())[:8]
        save_schedule_event_mapping(schedule_id, json.dumps(event))
        button = InlineKeyboardButton("Schedule Event", callback_data=f"schedule_event:{schedule_id}")
        reply_markup = InlineKeyboardMarkup([[button]])
        await query.edit_message_text(f"Summary:\n{summary}\n\nEvent detected:\n{event['title']} at {event['start']}\n\nWould you like to schedule it?", reply_markup=reply_markup)
    elif summary:
        await query.edit_message_text(f"Summary:\n{summary}")
    else:
        await query.edit_message_text("Sorry, I couldn't generate a summary for this email.")
    delete_short_id_mapping(short_id)

# --- Handler for the 'Send' button ---
async def send_email_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    chat_id = query.message.chat_id
    await query.answer()
    # Retrieve the stored draft and payload
    draft_info = email_draft_store.get(chat_id)
    if not draft_info:
        await query.edit_message_text("Sorry, I couldn't find the email draft to send.")
        return
    recipient = draft_info.get('recipient')
    subject = draft_info.get('subject')
    body = draft_info.get('body')
    if not recipient or not subject or not body:
        await query.edit_message_text("Sorry, the email draft is missing required information (recipient, subject, or body). Please try again.")
        return
    payload = {
        "recipient": recipient,
        "subject": subject,
        "body": body
    }
    logger.warning(f"[DEBUG] Sending email with payload: {payload}")
    try:
        response = requests.post(f"{MCP_SERVER_URL}/mcp/send_gmail", json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        await query.edit_message_text(f"✅ Email sent successfully!\n\n{result.get('result', '')}")
        email_draft_store.pop(chat_id, None)
    except Exception as e:
        logger.error(f"Error sending email: {e}")
        await query.edit_message_text(f"Sorry, there was an error sending the email: {e}")

def is_datetime_string(s):
    import re
    return bool(re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$', str(s).strip()))

async def schedule_event_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    chat_id = query.message.chat_id
    await query.answer()
    data = query.data
    schedule_id = None
    if data and data.startswith("schedule_event:"):
        schedule_id = data.split(":", 1)[1]
    import json
    event_json = get_schedule_event_mapping(schedule_id)
    if not schedule_id or not event_json:
        await query.edit_message_text("Sorry, I couldn't find the event details to schedule.")
        return
    event = json.loads(event_json)
    # Use as-is if already in correct format, else resolve
    if is_datetime_string(event.get("start")):
        start = event.get("start")
    else:
        start = resolve_natural_date(event.get("start"))
    if event.get("end") and is_datetime_string(event.get("end")):
        end = event.get("end")
    elif event.get("end"):
        end = resolve_natural_date(event.get("end"))
    else:
        end = None
    payload = {
        "summary": event.get("title"),
        "start": start,
        "end": end or start,
        "description": event.get("description"),
        "location": event.get("location"),
    }
    try:
        response = requests.post(f"{MCP_SERVER_URL}/mcp/create_calendar_event", json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        await query.edit_message_text(f"✅ Event scheduled in your calendar!\n\nTitle: {payload['summary']}\nStart: {payload['start']}\nEnd: {payload['end']}")
        delete_schedule_event_mapping(schedule_id)
    except Exception as e:
        logger.error(f"Error scheduling event: {e}")
        await query.edit_message_text(f"Sorry, there was an error scheduling the event: {e}")

def extract_email_from_user_input(user_input):
    # Simple regex to extract the first email address from the user input
    match = re.search(r"[\w\.-]+@[\w\.-]+", user_input)
    if match:
        return match.group(0)
    # Handle 'to myself' or 'to me' cases
    if re.search(r"to myself|to me|to my email|send me an email|ارسل لي|ارسل الى بريدي", user_input, re.IGNORECASE):
        return "hackathonfanar@gmail.com"
    return None

def main() -> None:
    """Start the bot."""
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not set in environment. Please add it to your .env file.")
        return

    # Initialize the database
    init_db()

    # Increase the timeout for the Telegram bot's HTTP requests.
    # Default is 5s, but image generation and upload can take longer.
    request = HTTPXRequest(
        connect_timeout=10.0,
        read_timeout=60.0, # Time to wait for a response from Telegram
        write_timeout=60.0 # Time to wait while sending data (e.g., uploading a photo)
    )

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).request(request).build()

    # --- Conversation Handler for Email ---
    email_handler = ConversationHandler(
        entry_points=[CommandHandler("email", email_command)],
        states={
            PROMPT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_prompt)],
            FILL_PLACEHOLDER: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_placeholder_input)],
            CONFIRM_SEND: [MessageHandler(filters.Regex(r"^(yes|no)$"), handle_confirmation)],
            AWAITING_RECIPIENT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_recipient)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    application.add_handler(email_handler)
    # --- Regular Handlers ---
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("generate", generate_image_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))
    application.add_handler(CallbackQueryHandler(summarize_email_callback, pattern="^summarize_email:"))
    application.add_handler(CallbackQueryHandler(send_email_callback, pattern="^send_email:"))
    application.add_handler(CallbackQueryHandler(schedule_event_callback, pattern="^schedule_event:"))

    # Run the bot until the user presses Ctrl-C
    logger.info("Starting Telegram bot...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)
    logger.info("Telegram bot stopped.")

if __name__ == "__main__":
    main() 