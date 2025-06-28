# Fanar AI Email Assistant

This project is a professional AI email assistant integrated with Telegram. Powered by Fanar's Large Language Models, it allows you to manage your Gmail and Google Calendar, perform web searches, and generate images directly from a Telegram chat.

## 🏗️ Architecture Overview

- The system is split into **two servers**:
  1. **Main MCP Server** (`mcp_server.py`): Handles all email, calendar, and agent logic (English/Arabic email agents, Gmail, Google Calendar, etc.).
  2. **Tools Server**: Handles auxiliary tools such as image generation and web search (if enabled/configured).
- The Telegram bot automatically routes requests to the correct server based on the tool or action required.

## Features

- **Compose and Send Emails**: Draft and send emails to any recipient through your Gmail account.
- **Generate Professional Emails**: Generate complete, professionally formatted emails in both English and Arabic from a simple prompt.
- **Google Calendar Integration**: Create and list calendar events.
- **Web Search**: Perform Google searches to get up-to-date information.
- **Image Generation**: Generate images from a text prompt using Fanar's ImageGen API.
- **Real-time Email Notifications**: Receive a Telegram notification whenever a new email arrives in your inbox.
- **Voice-to-Text & Text-to-Speech**: Interact with the bot using voice messages.

## ⚙️ Setup and Installation

### 1. Prerequisites
- Python 3.7+
- A Telegram account
- A Google account (for Gmail and Google Calendar)
- API keys for Fanar and SerpApi

### 2. Clone the Repository
```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 3. Set Up a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.
```bash
# Create the virtual environment
python -m venv venv

# Activate it
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 4. Install Dependencies
Install all the required Python packages from the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

## 🔑 Configuration

The application uses a `.env` file to manage all API keys and configuration variables.

1.  Create a file named `.env` in the root directory of the project.
2.  Add the following variables to the file, filling in your own values.

```dotenv
FANAR_API_KEY="fmFrMl3wHnB9SFnb8bzxNFpGCVE18Wcz"
MCP_SERVER_URL=http://127.0.0.1:8000
GMAIL_CLIENT_ID=462273424712-7u1lacesiad6hevcpscmc8oojp19evmo.apps.googleusercontent.com
GMAIL_CLIENT_SECRET=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
GMAIL_REDIRECT_URI=http://localhost:8080/
GMAIL_REFRESH_TOKEN=1//03c_7bnzOOGG6CgYIARAAGAMSNgF-L9Ir7zzcflOIzR2yypG742HKj4fOLFVFH2092Y4O_QsGubpSGTRoG1lse3oQmO2A3A7E5g
TELEGRAM_BOT_TOKEN=8070512628:AAEcrMtt7UuiODg2Aim13TIOdgcbEzVJM4g
SERPAPI_API_KEY=abcb75af623c08ee9665708617b6ed1e5c6b8491d1cee9eaa55839347f8291af
TELEGRAM_CHAT_ID=7081983767
WEBHOOK_URL=https://1270-86-36-20-121.ngrok-free.app/mcp/gmail_webhook
```

### 5. Generate Your Google Refresh Token

The application needs a refresh token to access your Gmail and Calendar data without asking you to log in every time. A script is provided to generate this.

1.  **Get Google Credentials**:
    *   Go to the [Google Cloud Console](https://console.cloud.google.com/).
    *   Create a new project.
    *   Enable the **Gmail API** and **Google Calendar API**.
    *   Go to "Credentials", create an "OAuth 2.0 Client ID" for a "Desktop app".
    *   Download the JSON file. Rename it to `credentials.json` and place it in the root of your project directory.

2.  **Run the Script**:
    *   Make sure you have filled in `GMAIL_CLIENT_ID` and `GMAIL_CLIENT_SECRET` in your `.env` file.
    *   Run the following command in your terminal:
        ```bash
        python get_gmail_refresh_token.py
        ```
    *   Your web browser will open. Log in with the Google account you want the bot to use.
    *   You will see a warning screen "Google hasn't verified this app". Click **"Advanced"** and proceed.
    *   Grant all the requested permissions (for sending mail, calendar, and modifying mail).
    *   The script will print a **Refresh Token** in your terminal.

3.  **Update `.env****:
    *   Copy the refresh token and paste it into the `GMAIL_REFRESH_TOKEN` variable in your `.env` file.

## 🚀 Running the Application

You need to run **two separate servers** (processes) in two different terminals.

### Terminal 1: Start the Main Backend Server
This server handles all the email, calendar, and agent logic.
```bash
uvicorn mcp_server:app --reload
```
You should see messages indicating that the server has started and that the Gmail polling thread is running.

### Terminal 2: Start the Tools Server (if using image generation, web search, etc.)
If you have a separate tools server (for image generation, web search, etc.), start it according to its own instructions (e.g., `uvicorn mcp_server_tools:app --reload` or similar).

### Terminal 3: Start the Telegram Bot
This process listens for messages from your Telegram chat and routes requests to the correct server.
```bash
python telegram_bot.py
```

- The bot will automatically send email/calendar/agent requests to the main MCP server, and tool requests (image generation, web search) to the tools server.

## 📨 Email Draft Approval Workflow

- When you request an email, the bot will generate a professional draft (in English or Arabic) using the appropriate agent.
- The draft is shown to you in Telegram, along with a **Send** button.
- The email is **only sent if you click 'Send'**. This ensures you always review the draft before it is sent.
- The agents will:
  - Only use a generic salutation ("Dear," or "عزيزي،") unless you explicitly provide a recipient name.
  - Never guess or invent recipient names from email addresses or context.
  - Never add explanations, disclaimers, or meta-commentary.
  - Always include a subject line at the top of the email ("Subject: ..." or "موضوع: ...").
  - Never invent or elaborate on your request—only restate, rephrase, and format your main message as a professional email.

## 🤖 How to Use the Bot

Once both processes are running, open Telegram and start a chat with your bot.

### Example Commands:
- **Send an email draft for review**:
  > `send an email to example@email.com with the subject "Meeting" and the body "Hi, are you available tomorrow at 2pm?"`
- **Send an email to yourself**:
  > `send me an email to myself with the title "Test" and body "This is a test"`
- **Send an email without a recipient name** (uses generic salutation):
  > `send an email to danijeun@gmail.com asking what's for dinner?`
- **Generate a professional Arabic email**:
  > `generate an arabic email about requesting a meeting next week to discuss the project proposal`
- **Create a calendar event**:
  > `create an event for tomorrow at 4pm called "Project Sync-up"`
- **Search the web**:
  > `what is the weather like in Doha today?`
- **Generate an image**:
  > `/generate a photorealistic image of a cat programming on a laptop`
- **Use your voice**: Send a voice message and the bot will transcribe it, process it, and reply with a voice message.
- **Receive email notifications**: Simply wait. When a new email arrives in the configured Gmail account, you will receive a notification automatically.

## 🛠️ Troubleshooting

- **Missing Subject**: The system always enforces a subject line. If the LLM omits it, a default subject is added automatically.
- **Missing Recipient**: If you say "send to myself" or similar, the recipient is set to `hackathonfanar@gmail.com`. Otherwise, the bot extracts the recipient from your message.
- **Draft Not Sent**: Emails are only sent after you click the 'Send' button in Telegram. If you do not see the button, check your conversation history or try again.
- **Agent Adds Extra Content**: The agents are strictly instructed to never add explanations, disclaimers, or invent content. If you see this, please update the system or contact the maintainer.
- **Server Not Responding**: Make sure both the main MCP server and the tools server (if used) are running and accessible at the URLs configured in your `.env` file. 