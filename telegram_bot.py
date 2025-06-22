import os
import io
import asyncio
from dotenv import load_dotenv
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.request import HTTPXRequest
from backend.app import get_agent_response, get_vision_response, generate_image_from_prompt
from backend.database import init_db, get_conversation_history, update_conversation_history, clear_conversation_history
import nest_asyncio
from openai import OpenAI
from pydub import AudioSegment

# Apply nest_asyncio to allow asyncio to be nested, which is needed by the telegram library
nest_asyncio.apply()

# --- Setup ---
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
FANAR_API_KEY = os.getenv("FANAR_API_KEY")

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

# --- Bot Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message and asks the user to log in."""
    user = update.effective_user
    chat_id = update.message.chat_id
    clear_conversation_history(chat_id)  # Reset history on start

    await update.message.reply_html(
        rf"Hi {user.mention_html()}! I'm your professional AI email assistant, powered by Fanar. "
        "I can help you compose, format, and send emails to anyone from hackathonfanar@gmail.com. "
        "To get started, just tell me what email you'd like to send."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles all incoming text and photo messages."""
    user_id = update.message.from_user.id
    chat_id = update.message.chat_id
    user_input = update.message.text
    
    # Get user's conversation history from the database
    history = get_conversation_history(chat_id)
    
    # Send a "typing..." notification
    await context.bot.send_chat_action(chat_id=chat_id, action='typing')
    
    # Get the agent's response
    # Since get_agent_response is not an async function, we run it in a thread
    loop = asyncio.get_event_loop()
    text_response, media_response, updated_history = await loop.run_in_executor(
        None, get_agent_response, user_input, history
    )

    # Update the user's history in the database
    update_conversation_history(chat_id, updated_history)
    
    # Send the response back to the user
    if media_response:
        # If there's an image, send it with the text as a caption
        await update.message.reply_photo(photo=media_response, caption=text_response)
    elif text_response:
        # Otherwise, send the text response
        await update.message.reply_text(text_response)

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
            loop = asyncio.get_event_loop()
            text_response, media_response, updated_history = await loop.run_in_executor(
                None, get_agent_response, transcribed_text, history
            )
            update_conversation_history(chat_id, updated_history)
            
            # Handle the response (could be text or an image)
            if media_response:
                # If there's an image, send it with the text as a caption
                await update.message.reply_photo(photo=media_response, caption=text_response)
            elif text_response:
                # Otherwise, convert text to speech and send as audio
                audio_data = get_tts_audio(text_response)
                if audio_data:
                    await update.message.reply_audio(audio=audio_data, title="Fanar Response")
                else:
                    await update.message.reply_text("Sorry, I had an issue generating the audio response, but here is my answer in text:\n\n" + text_response)
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

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))

    # Run the bot until the user presses Ctrl-C
    logger.info("Starting Telegram bot...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)
    logger.info("Telegram bot stopped.")

if __name__ == "__main__":
    main() 