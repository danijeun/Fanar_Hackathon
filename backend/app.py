import json
import os
from dotenv import load_dotenv
from openai import OpenAI
import re
from datetime import datetime, timedelta
import pytz
from tzlocal import get_localzone
from backend.prompts import SYSTEM_PROMPTS, CLASSIFICATION_PROMPT, MASTER_SYSTEM_PROMPT
from backend.mcp_client import MCPClient
import base64
import requests

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
    conversation_history.append({"role": "user", "content": user_input})
    mcp_client = MCPClient()

    try:
        # Step 1: Prepare messages for the master agent
        system_prompt = MASTER_SYSTEM_PROMPT.replace("{{current_date}}", datetime.now().strftime('%Y-%m-%d'))
        
        # Use a recent segment of the conversation history for context
        messages = [
            {"role": "system", "content": system_prompt},
            *conversation_history[-7:]
        ]

        # Step 2: Get the agent's response (which could be text or a tool call)
        agent_response_text = get_completion(messages)
        conversation_history.append({"role": "assistant", "content": agent_response_text})
        
        # Step 3: Check for tool calls in the agent's response
        final_text = agent_response_text
        final_media = None
        
        try:
            # Use regex to find all JSON code blocks (for single or multiple tool calls)
            json_blocks = re.findall(r'```json\s*(\{.*?\})\s*```', agent_response_text, re.DOTALL)
            
            if json_blocks:
                all_results = []
                parsed_tool_calls = [json.loads(block) for block in json_blocks]

                # --- Pre-processing Step ---
                last_user_message = ""
                for message in reversed(conversation_history):
                    if message['role'] == 'user':
                        last_user_message = message['content'].lower()
                        break
                
                for call in parsed_tool_calls:
                    payload = call.get("payload", {})
                    if "recipient" in payload and ("<" in payload["recipient"] or "placeholder" in payload["recipient"]):
                        if _is_self_send_request(last_user_message):
                            payload["recipient"] = "hackathonfanar@gmail.com"
                        else:
                            email_regex = r'[\w\.-]+@[\w\.-]+'
                            match = re.search(email_regex, last_user_message)
                            if match:
                                payload["recipient"] = match.group(0)

                # Loop through each found JSON block string
                for tool_data in parsed_tool_calls:
                    try:
                        tool_name = tool_data.get("tool")
                        payload = tool_data.get("payload", {})

                        # Pre-process payload for calendar tools if needed
                        if tool_name in ["create_calendar_event", "list_calendar_events"]:
                            for key in ['start', 'end', 'when']:
                                if key in payload and isinstance(payload[key], str):
                                    time_str = payload[key]
                                    if time_str in ["today", "tomorrow"]:
                                        if tool_name == "list_calendar_events":
                                            start, end = get_utc_iso_range(time_str)
                                            payload['timeMin'], payload['timeMax'] = start, end
                                    else:
                                        dt_obj = parse_special_time_format(time_str)
                                        if dt_obj:
                                            payload[key] = dt_obj.strftime("%Y-%m-%dT%H:%M:%SZ")

                        # --- Execute the tool ---
                        result = mcp_client.execute_tool(tool_name, payload)
                        all_results.append({"tool": tool_name, "result": result})

                        # If this is an image, we need to handle the media response immediately
                        if tool_name == "generate_image" and isinstance(result, dict) and "image_b64" in result:
                            final_media = base64.b64decode(result["image_b64"])
                    
                    except (KeyError) as e:
                        all_results.append({"tool": "unknown", "result": {"error": f"Invalid tool format: {e}"}})

                # --- Summarize all results ---
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
                
                summary_prompt = (
                    "You are a summarizer. Your ONLY job is to respond with a single, short, user-facing sentence. "
                    "The outcome of the actions is in the 'TOOL_RESULTS' system message. "
                    "Do NOT add any details from the tools' results. Do NOT show JSON, do NOT mention the tools by name, and do NOT repeat the user's request. "
                    "If the 'TOOL_RESULTS' contains an 'error', your response MUST be 'I'm sorry, the request failed because of an error.' "
                    "If there is no error, your response MUST be a simple confirmation like 'Done.' or 'I've taken care of that for you.' "
                    "Be as brief as possible. Now, summarize the preceding 'TOOL_RESULTS'."
                )
                summary_messages = [
                    {"role": "system", "content": system_prompt},
                    *conversation_history[-8:],
                    {"role": "user", "content": summary_prompt}
                ]
                final_text = get_completion(summary_messages)
                conversation_history.append({"role": "assistant", "content": final_text})

        except Exception as e:
            final_text = "I'm sorry, there was an error processing the tool request."
        
        return final_text, final_media, conversation_history

    except Exception as e:
        return "I'm sorry, a critical error occurred. Please try your request again.", None, conversation_history

def generate_image_from_prompt(prompt: str):
    """
    Generates an image from a text prompt by calling the MCP server.
    Returns the image as bytes, or None if an error occurs.
    """
    try:
        mcp_client = MCPClient()
        payload = {"prompt": prompt}
        result = mcp_client.call("generate_image", payload)
        
        if result and "image_b64" in result:
            image_b64 = result["image_b64"]
            image_bytes = base64.b64decode(image_b64)
            return image_bytes
            
        return None
        
    except Exception as e:
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
    """Check if the user is asking to send something to themselves."""
    # Simple check for phrases like "send it to me" or "email me"
    self_email_phrases = ["send me", "to me", "to myself", "my email"]
    return any(phrase in user_message for phrase in self_email_phrases)

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
