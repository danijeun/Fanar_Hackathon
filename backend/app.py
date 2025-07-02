import json
import os
from dotenv import load_dotenv
from openai import OpenAI
import re
from datetime import datetime, timedelta
import pytz
from tzlocal import get_localzone
from backend.prompts import MASTER_SYSTEM_PROMPT
from backend.mcp_client import MCPClient
import base64
import requests
from typing import List, Dict
from backend.utils import resolve_natural_date  # Import for fallback date resolution
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_utc_iso_range(when="today"):
    """
    Get UTC ISO formatted datetime range for 'today' or 'tomorrow'.
    Returns a tuple of (start_time, end_time) in UTC ISO format.
    """
    local_tz = get_localzone()
    now = datetime.now(local_tz)
    
    if when == "tomorrow":
        target_date = now.date() + timedelta(days=1)
    else:  # today
        target_date = now.date()
    
    # Create datetime objects for start and end of the day in local time
    start_dt = datetime.combine(target_date, datetime.min.time()).replace(tzinfo=local_tz)
    end_dt = datetime.combine(target_date + timedelta(days=1), datetime.min.time()).replace(tzinfo=local_tz)
    
    # Convert to UTC
    start_utc = start_dt.astimezone(pytz.UTC)
    end_utc = end_dt.astimezone(pytz.UTC)
    
    return (
        start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        end_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    )

def local_to_utc_iso(local_dt_str, local_fmt="%Y-%m-%d %H:%M"):
    """
    Convert a local datetime string to UTC ISO format for Google Calendar.
    local_dt_str: e.g. '2024-06-20 15:00'
    local_fmt: format of the input string
    Returns: ISO string in UTC, e.g. '2024-06-20T13:00:00Z'
    """
    try:
        # Parse the local datetime string
        local_dt = datetime.strptime(local_dt_str, local_fmt)
        
        # Get the local timezone
        local_tz = get_localzone()
        
        # Make the datetime timezone-aware using the local timezone
        if hasattr(local_tz, 'localize'):  # pytz timezone
            local_aware = local_tz.localize(local_dt)
        else:  # zoneinfo timezone
            local_aware = local_dt.replace(tzinfo=local_tz)
        
        # Convert to UTC
        utc_dt = local_aware.astimezone(pytz.UTC)
        return utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception as e:
        print(f"[DEBUG] Error in local_to_utc_iso: {str(e)}")
        raise

def get_message_category(user_input):
    """
    DEPRECATED: This function is no longer used with the master prompt architecture.
    """
    return None

def get_system_prompt(user_input):
    """
    DEPRECATED: This function is no longer used with the master prompt architecture.
    """
    return MASTER_SYSTEM_PROMPT

def parse_special_time_format(time_str):
    """
    Parse special time formats like 'todayT19:00:00' or 'tomorrowT20:00:00'
    Returns a datetime object
    """
    if not time_str or not time_str.startswith(('today', 'tomorrow')):
        return None
    parts = time_str.split('T')
    if len(parts) != 2: return None
    base_date = datetime.now() + timedelta(days=1) if time_str.startswith('tomorrow') else datetime.now()
    try:
        time_components = parts[1].replace('Z', '').split(':')
        if len(time_components) == 2: # HH:MM
            hour, minute = map(int, time_components)
            second = 0
        elif len(time_components) == 3: # HH:MM:SS
            hour, minute, second = map(int, time_components)
        else:
            return None
            
        result = base_date.replace(hour=hour, minute=minute, second=second, microsecond=0)
        local_tz = get_localzone()
        return local_tz.localize(result) if hasattr(local_tz, 'localize') else result.replace(tzinfo=local_tz)
    except (ValueError, IndexError) as e:
        print(f"Error parsing special time format '{time_str}': {e}")
        return None

# Load API key
load_dotenv()
client = OpenAI(
    base_url="https://api.fanar.qa/v1",
    api_key=os.getenv("FANAR_API_KEY"),
)
model_name = "Fanar"

def get_completion(messages):
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
    )
    content = response.choices[0].message.content
    return content

def get_agent_response(user_input, conversation_history):
    logging.info(f"User input: {user_input}")
    conversation_history.append({"role": "user", "content": user_input})
    mcp_client = MCPClient()

    try:
        # Step 1: Prepare messages for the master agent
        system_prompt = MASTER_SYSTEM_PROMPT.replace("{{current_date}}", datetime.now().strftime('%Y-%m-%d'))
        
        # Use a recent segment of the conversation history for context (limit to last 3 messages, truncate each to 1000 chars)
        last_msgs = conversation_history[-3:]
        truncated_msgs = []
        for msg in last_msgs:
            truncated_msg = msg.copy()
            if len(truncated_msg["content"]) > 1000:
                truncated_msg["content"] = truncated_msg["content"][:1000]
            truncated_msgs.append(truncated_msg)
        # Check total context length
        total_context = system_prompt + ''.join([m["content"] for m in truncated_msgs])
        if len(total_context) > 12000:
            # Summarize the last 3 messages into one
            summary = summarize_messages_with_llm(truncated_msgs)
            truncated_msgs = [{"role": "user", "content": f"SUMMARY_OF_LAST_MESSAGES: {summary}"}]
        messages = [
            {"role": "system", "content": system_prompt},
            *truncated_msgs
        ]

        logging.info(f"Messages sent to agent: {messages}")

        # Step 2: Get the agent's response (which could be text or a tool call)
        print("\n--- 1. Agent is thinking... ---")
        agent_response_text = get_completion(messages)
        logging.info(f"Agent's raw response: {agent_response_text}")
        print(f"--- 2. Agent's raw response: ---\n{agent_response_text}\n--------------------")
        conversation_history.append({"role": "assistant", "content": agent_response_text})
        
        # Step 3: Check for tool calls in the agent's response
        final_text = agent_response_text
        final_media = None
        
        try:
            # Use regex to find all JSON code blocks (for single or multiple tool calls)
            json_blocks = re.findall(r'```json\s*([\s\S]*?)\s*```', agent_response_text)
            
            if not json_blocks:
                logging.info(f"Final response to UI (no tool call): {agent_response_text}")
                return agent_response_text, None, conversation_history
            
            if json_blocks:
                all_results = []
                parsed_tool_calls = []
                for block in json_blocks:
                    try:
                        cleaned = clean_json_block(block)
                        try:
                            parsed = json.loads(cleaned)
                        except json.JSONDecodeError as e:
                            # Try to wrap in [] and parse as a list
                            try:
                                parsed = json.loads(f'[{cleaned}]')
                            except Exception as e2:
                                logging.error(f"Failed to parse tool call JSON after cleaning (even as list): {block}\nError: {e2}")
                                continue
                        # If it's a string, treat as direct assistant response
                        if isinstance(parsed, str):
                            logging.info(f"Returning direct assistant response (string): {parsed}")
                            return parsed, None, conversation_history
                        # If it's a dict with only a 'response' key, treat as direct assistant response
                        if isinstance(parsed, dict) and set(parsed.keys()) == {'response'}:
                            logging.info(f"Returning direct assistant response (response): {parsed['response']}")
                            return parsed['response'], None, conversation_history
                        # If it's a dict with only a 'message' key, treat as direct assistant response
                        if isinstance(parsed, dict) and set(parsed.keys()) == {'message'}:
                            logging.info(f"Returning direct assistant response (message): {parsed['message']}")
                            return parsed['message'], None, conversation_history
                        # If it's a dict with 'actions', treat each as a tool call
                        if isinstance(parsed, dict) and 'actions' in parsed and isinstance(parsed['actions'], list):
                            parsed_tool_calls.extend(parsed['actions'])
                        # If it's a list, treat each as a tool call
                        elif isinstance(parsed, list):
                            parsed_tool_calls.extend(parsed)
                        # If it's a dict, treat as a single tool call
                        elif isinstance(parsed, dict):
                            parsed_tool_calls.append(parsed)
                        else:
                            logging.error(f"Unrecognized tool call JSON structure: {parsed}")
                    except Exception as e:
                        logging.error(f"Failed to parse tool call JSON after cleaning: {block}\nError: {e}")
                        continue
                logging.info(f"Parsed tool calls: {parsed_tool_calls}")

                # --- Pre-processing Step ---
                last_user_message = ""
                for message in reversed(conversation_history):
                    if message['role'] == 'user':
                        last_user_message = message['content'].lower()
                        break
                
                for call in parsed_tool_calls:
                    payload = call.get("payload", {})
                    # This check is crucial. We look for generic placeholders or requests to send to oneself.
                    if "recipient" in payload and ("<" in payload["recipient"] or "placeholder" in payload["recipient"] or _is_self_send_request(last_user_message)):
                        print("--- SELF-SEND or PLACEHOLDER DETECTED. Overriding recipient. ---")
                        payload["recipient"] = "hackathonfanar@gmail.com"
                        call["payload"] = payload
                    # If no valid recipient is found, try to extract one from the user's message
                    elif "recipient" in payload:
                        email_regex = r'[\w\.-]+@[\w\.-]+'
                        match = re.search(email_regex, last_user_message)
                        if match:
                            payload["recipient"] = match.group(0)
                            call["payload"] = payload
                    # --- PATCH: Ensure recipient_email is present for email agents ---
                    if call.get("tool") in ("arabic_email_agent", "english_email_agent"):
                        if ("recipient_email" not in payload or not payload["recipient_email"]) or _is_self_send_request(last_user_message):
                            # Try to extract from user message or override for self-send
                            if _is_self_send_request(last_user_message):
                                payload["recipient_email"] = "hackathonfanar@gmail.com"
                                call["payload"] = payload
                            else:
                                email_regex = r'[\w\.-]+@[\w\.-]+'
                                match = re.search(email_regex, last_user_message)
                                if match:
                                    payload["recipient_email"] = match.group(0)
                                    call["payload"] = payload
                                else:
                                    # If recipient email is truly missing and cannot be inferred, block execution
                                    return ("Please provide the recipient's email address to send the email.", None, conversation_history)

                # --- STRICT ARABIC EMAIL WORKFLOW ENFORCEMENT ---
                # If any tool call is send_gmail and the body is Arabic, enforce the full chain
                def is_arabic_text(text):
                    return any('\u0600' <= c <= '\u06FF' for c in text)

                for idx, tool_data in enumerate(parsed_tool_calls):
                    if tool_data.get("tool") == "send_gmail":
                        body = tool_data.get("payload", {}).get("body", "")
                        # Only enforce for Arabic emails
                        if is_arabic_text(body):
                            # Look for the required chain: format_professional_arabic_email -> send_gmail
                            found_format = False
                            fmt_idx = -1
                            for i, t in enumerate(parsed_tool_calls[:idx]):
                                if t.get("tool") == "format_professional_arabic_email":
                                    found_format = True
                                    fmt_idx = i
                            if not (found_format and fmt_idx < idx):
                                # Block send_gmail and return error
                                tool_data["payload"]["body"] = "[ERROR: You must use the following tool chain for Arabic emails: format_professional_arabic_email -> send_gmail. Please revise your tool call sequence.]"
                                continue
                # Now execute all tools in order (with fixed send_gmail)
                for tool_data in parsed_tool_calls:
                    try:
                        tool_name = tool_data.get("tool")
                        payload = tool_data.get("payload", {})
                        logging.info(f"Executing tool: {tool_name} with payload: {payload}")
                        result = mcp_client.execute_tool(tool_name, payload)
                        logging.info(f"Tool result for {tool_name}: {result}")
                        all_results.append({"tool": tool_name, "result": result})
                        # If this is an image, we need to handle the media response immediately
                        if tool_name == "generate_image" and isinstance(result, dict) and "image_b64" in result:
                            final_media = base64.b64decode(result["image_b64"])
                    except (KeyError) as e:
                        all_results.append({"tool": "unknown", "result": {"error": f"Invalid tool format: {e}"}})

                # --- Summarize all results (improved logic) ---
                # Sanitize image data before adding to history
                sanitized_results = []
                for res in all_results:
                    if res.get("tool") == "generate_image" and "image_b64" in res.get("result", {}):
                        s_res = res.copy()
                        s_res["result"] = s_res["result"].copy()
                        s_res["result"]["image_b64"] = "[Image data was generated successfully]"
                        sanitized_results.append(s_res)
                    else:
                        sanitized_results.append(res)
                
                conversation_history.append({"role": "system", "content": f"TOOL_RESULTS: {json.dumps(sanitized_results)}"})
                logging.info(f"Tool results appended to history: {sanitized_results}")

                # Improved summary logic
                error_messages = []
                success_messages = []
                for idx, res in enumerate(sanitized_results):
                    tool = res.get("tool")
                    result = res.get("result", {})
                    if "error" in result:
                        error_messages.append(f"{tool}: {result['error']}")
                    else:
                        # Custom success messages per tool
                        if tool == "send_gmail":
                            success_messages.append("Your email was sent successfully.")
                        elif tool == "create_calendar_event":
                            success_messages.append("Your calendar event was created successfully.")
                        elif tool == "list_calendar_events":
                            events = result.get("events", [])
                            if not events:
                                success_messages.append("You have no events for the selected period.")
                            else:
                                event_lines = []
                                for event in events:
                                    summary = event.get("summary", "No Title")
                                    start = event.get("start", {}).get("dateTime") or event.get("start", {}).get("date")
                                    end = event.get("end", {}).get("dateTime") or event.get("end", {}).get("date")
                                    event_lines.append(f"- {summary} ({start} to {end})")
                                success_messages.append("Here are your calendar events:\n" + "\n".join(event_lines))
                        elif tool == "generate_image":
                            success_messages.append("The image was generated successfully.")
                        elif tool == "web_search":
                            results = result.get("results", [])
                            if results:
                                snippets = []
                                for item in results:
                                    snippet = item.get("snippet")
                                    title = item.get("title")
                                    if snippet:
                                        snippets.append(snippet.strip())
                                    elif title:
                                        snippets.append(title)
                                if snippets:
                                    summary_prompt = (
                                        "Summarize the following web search results in a clear and concise way for the user. "
                                        "Focus on the main findings and avoid repetition.\n\nResults:\n" + "\n".join(snippets)
                                    )
                                    try:
                                        summary_response = client.chat.completions.create(
                                            model=model_name,
                                            messages=[{"role": "system", "content": "You are a helpful assistant that summarizes web search results."}, {"role": "user", "content": summary_prompt}],
                                            max_tokens=128
                                        )
                                        summary = summary_response.choices[0].message.content.strip()
                                        success_messages.append(summary)
                                    except Exception as e:
                                        print(f"[ERROR] LLM web search summarization failed: {e}")
                                        success_messages.append("Web search results: " + " ".join(snippets))
                                else:
                                    success_messages.append("No readable web search summaries found.")
                            else:
                                success_messages.append("No web search results found.")
                        else:
                            success_messages.append(f"{tool} completed successfully.")
                if error_messages:
                    non_json_text = re.sub(r'```json.*?```', '', agent_response_text, flags=re.DOTALL).strip()
                    if non_json_text:
                        final_text = f"{non_json_text}\n\n[Tool Error(s)]:\n" + "\n".join(error_messages)
                    else:
                        final_text = "[Tool Error(s)]:\n" + "\n".join(error_messages)
                elif success_messages:
                    final_text = "\n".join(success_messages)
                else:
                    final_text = agent_response_text
                logging.info(f"Final response to UI: {final_text}")
                conversation_history.append({"role": "assistant", "content": final_text})

        except Exception as e:
            final_text = "I'm sorry, there was an error processing the tool request."
            logging.error(f"Exception in tool processing: {e}")
        
        return final_text, final_media, conversation_history

    except Exception as e:
        logging.error(f"Critical error in get_agent_response: {e}")
        return "I'm sorry, a critical error occurred. Please try your request again.", None, conversation_history

def generate_image_from_prompt(prompt: str):
    """
    Generates an image from a text prompt by calling the MCP server's /mcp/generate_image endpoint.
    Returns the image as bytes, or None if an error occurs.
    """
    try:
        # Use the MCP server endpoint directly for image generation
        mcp_url = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000")
        url = f"{mcp_url}/mcp/generate_image"
        payload = {"prompt": prompt}
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        image_b64 = result.get("image_b64")
        if not image_b64:
            return None
        image_bytes = base64.b64decode(image_b64)
        return image_bytes
    except Exception as e:
        print(f"[ERROR] generate_image_from_prompt: {e}")
        return None

def get_vision_response(prompt: str, image_bytes: bytes):
    """
    Gets a description for an image using the Fanar Vision API.
    """
    headers = {
        "Authorization": f"Bearer {os.getenv('FANAR_API_KEY')}",
        "Content-Type": "application/json",
    }
    
    # Encode the image to base64
    raw_b64 = base64.b64encode(image_bytes).decode("utf-8")
    image_b64_url = f"data:image/jpeg;base64,{raw_b64}"
    
    # Create the payload
    payload = {
        "model": "Fanar-Oryx-IVU-1",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_b64_url
                        }
                    }
                ]
            }
        ],
        "max_tokens": 750,
    }
    
    try:
        response = requests.post("https://api.fanar.qa/v1/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        api_response = response.json()
        content = api_response["choices"][0]["message"]["content"]
        return content
    except requests.exceptions.RequestException as e:
        return f"Sorry, there was an error calling the vision API: {e}"
    except Exception as e:
        return "Sorry, an unexpected error occurred while processing the image."

def _find_recipient_in_tool_calls(tool_calls: list[dict]) -> str | None:
    """Finds the 'recipient' value in a list of tool calls."""
    for call in tool_calls:
        if "payload" in call and "recipient" in call["payload"]:
            return call["payload"]["recipient"]
    return None

def _is_self_send_request(user_message: str) -> bool:
    """Check if the user is asking to send an email to themselves."""
    self_phrases = ["to me", "to myself", "send me", "send it to me"]
    return any(phrase in user_message.lower() for phrase in self_phrases)

def summarize_messages_with_llm(messages: List[Dict]) -> str:
    """
    Uses the Fanar LLM to summarize a list of conversation messages into a concise summary.
    """
    if not messages:
        return ""
    # Only use the last 3 messages for summarization context, truncate each to 1000 chars
    messages = messages[-3:]
    for msg in messages:
        if len(msg["content"]) > 1000:
            msg["content"] = msg["content"][:1000]
    # Build a prompt for summarization
    conversation_text = "\n".join([
        f"User: {msg['content']}" if msg['role'] == 'user' else f"Assistant: {msg['content']}" if msg['role'] == 'assistant' else f"System: {msg['content']}" for msg in messages
    ])
    prompt = (
        "Summarize the following conversation history in a concise way, focusing on the most important requests, actions, and results. "
        "Do not include irrelevant details.\n\n"
        f"Conversation:\n{conversation_text}\n\nSummary:"
    )
    # Use the Fanar LLM to get the summary
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "system", "content": "You are a helpful assistant that summarizes conversations."}, {"role": "user", "content": prompt}],
        max_tokens=128
    )
    summary = response.choices[0].message.content.strip()
    return summary

def summarize_conversation_history(history: List[Dict]) -> List[Dict]:
    """
    Summarizes the conversation history to keep only the most important information.
    This function keeps the last 5 messages, but for older messages, it replaces them with a single LLM-generated summary message.
    """
    if len(history) <= 5:
        return history
    # Summarize all but the last 5 messages using the LLM
    summary = summarize_messages_with_llm(history[:-5])
    summarized = [{"role": "system", "content": f"SUMMARY_OF_PREVIOUS: {summary}"}]
    return summarized + history[-5:]

def update_conversation_history(chat_id, history):
    # Remove tool results from history before saving
    filtered_history = []
    for msg in history:
        if isinstance(msg, dict) and msg.get('role') == 'system' and 'TOOL_RESULTS' in msg.get('content', ''):
            continue  # Skip tool results
        filtered_history.append(msg)
    # Save only filtered history
    # ... existing code to save filtered_history instead of history ...

def extract_event_from_text(text: str):
    """
    Uses the Fanar LLM to extract event details from a block of text (e.g., an email body).
    Returns a dict with keys: title, start, end, location, description if an event is found, else None.
    Handles natural language dates like 'today', 'tomorrow', etc.
    """
    print(f"[EVENT EXTRACTION] Analyzing text: {text}")
    from datetime import datetime
    now = datetime.now()
    current_date = now.strftime('%Y-%m-%d')
    current_time = now.strftime('%H:%M')
    prompt = (
        f"Today is {current_date}. The current time is {current_time}.\n"
        "You are an assistant that reads emails and extracts event details.\n"
        "Read the following email. If it contains an event (meeting, dinner, hackathon, etc.), extract:\n"
        "- title: The main topic, subject, or purpose of the event (e.g., 'Hackathon', 'Dinner with Bob').\n"
        "- start: The start date and time, in the format YYYY-MM-DD HH:MM (e.g., '2025-06-28 17:00').\n"
        "- end: The end date and time, in the same format. If only a start time is given, set end to 1 hour after start.\n"
        "- location: The location if mentioned, else null.\n"
        "- description: A short summary of the event.\n"
        "Return ONLY a JSON object with these keys. If you can't find a title, use the main subject or topic of the email.\n"
        "If there is no event, return the string 'NO_EVENT'.\n\n"
        f"Email:\n{text}\n\nEvent details:"
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "system", "content": "You extract event details from text."}, {"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.2
    )
    content = response.choices[0].message.content.strip()
    print(f"[EVENT EXTRACTION] LLM raw output: {content}")
    if content.strip().upper().startswith("NO_EVENT"):
        print(f"[EVENT EXTRACTION] Result: None (NO_EVENT)")
        return None
    try:
        event = json.loads(content)
        print(f"[EVENT EXTRACTION] LLM parsed JSON: {event}")
        # Basic validation
        if not event.get("title") or not event.get("start"):
            print(f"[EVENT EXTRACTION] Result: None (missing title/start)")
            return None
        # Post-process: if end is missing, assume 1 hour after start
        from datetime import datetime, timedelta
        def to_dt(s):
            try:
                return datetime.strptime(s, "%Y-%m-%d %H:%M")
            except Exception:
                return None
        start_dt = to_dt(event['start'])
        if start_dt:
            if not event.get('end') or not to_dt(event['end']):
                end_dt = start_dt + timedelta(hours=1)
                event['end'] = end_dt.strftime("%Y-%m-%d %H:%M")
                print(f"[EVENT EXTRACTION] Defaulted end time to one hour after start: {event['end']}")
            else:
                # Ensure end is formatted correctly
                event['end'] = to_dt(event['end']).strftime("%Y-%m-%d %H:%M")
            # Ensure start is formatted correctly
            event['start'] = start_dt.strftime("%Y-%m-%d %H:%M")
        print(f"[EVENT EXTRACTION] Result: {event}")
        return event
    except Exception as e:
        print(f"[EVENT EXTRACTION] LLM output not valid JSON: {e}")
        return None

def clean_json_block(block):
    # Remove comments starting with # or // (inline or full line)
    block = re.sub(r'(?m)\s*#.*$', '', block)  # Remove # comments
    block = re.sub(r'(?m)\s*//.*$', '', block)  # Remove // comments
    # Remove comments after commas or on the same line
    block = re.sub(r',\s*#.*$', ',', block, flags=re.MULTILINE)
    block = re.sub(r',\s*//.*$', ',', block, flags=re.MULTILINE)
    block = re.sub(r'\s+#.*$', '', block, flags=re.MULTILINE)
    block = re.sub(r'\s+//.*$', '', block, flags=re.MULTILINE)
    # Replace single quotes with double quotes
    block = re.sub(r"'", '"', block)
    # Remove trailing commas before } or ]
    block = re.sub(r',\s*([}\]])', r'\1', block)
    # Remove any blank lines
    block = '\n'.join([line for line in block.splitlines() if line.strip()])
    return block

def main():
    load_dotenv()
    
    conversation_history = []
    
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() == 'exit':
            break
            
        final_response, media, updated_history = get_agent_response(user_input, conversation_history)
        conversation_history = updated_history
        print("Assistant:", final_response)
        if media:
            print("[Image was generated, but cannot be displayed in CLI]")

if __name__ == "__main__":
    main()