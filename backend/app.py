import json
import os
from dotenv import load_dotenv
from openai import OpenAI
import re
from datetime import datetime, timedelta
import pytz
from tzlocal import get_localzone
from backend.prompts import SYSTEM_PROMPTS, CLASSIFICATION_PROMPT
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
    Use Fanar LLM to classify the user's message into a category.
    Returns the category as a string: 'calendar', 'translation', 'email', or 'general'
    """
    messages = [
        {"role": "system", "content": CLASSIFICATION_PROMPT},
        {"role": "user", "content": user_input}
    ]
    
    try:
        category = get_completion(messages).strip().lower()
        print(f"[DEBUG] Message category: {category}")
        return category if category in ['calendar', 'translation', 'email'] else 'general'
    except Exception as e:
        print(f"[DEBUG] Classification error: {e}")
        return 'general'

def get_system_prompt(user_input):
    """
    Use Fanar LLM to classify the message and return the appropriate system prompt.
    """
    category = get_message_category(user_input)
    return SYSTEM_PROMPTS[category]

def parse_special_time_format(time_str):
    """
    Parse special time formats like 'todayT19:00:00' or 'tomorrowT20:00:00'
    Returns a datetime object
    """
    if not time_str:
        return None
        
    # Handle special formats
    if time_str.startswith(('today', 'tomorrow')):
        # Extract the time part after 'T'
        parts = time_str.split('T')
        if len(parts) != 2:
            raise ValueError(f"Invalid time format: {time_str}")
            
        time_part = parts[1].replace('Z', '')  # Remove Z if present
        if not ':' in time_part:
            time_part = f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:]}"
            
        # Create base date
        base_date = datetime.now()
        if time_str.startswith('tomorrow'):
            base_date += timedelta(days=1)
            
        # Parse time components
        try:
            hour, minute, second = map(int, time_part.split(':'))
            result = base_date.replace(hour=hour, minute=minute, second=second, microsecond=0)
            
            # Convert to local timezone
            local_tz = get_localzone()
            if hasattr(local_tz, 'localize'):
                result = local_tz.localize(result)
            else:
                result = result.replace(tzinfo=local_tz)
                
            return result
            
        except ValueError as e:
            raise ValueError(f"Invalid time format in {time_str}: {e}")
            
    return None

# Load API key
load_dotenv()
client = OpenAI(
    base_url="https://api.fanar.qa/v1",
    api_key=os.getenv("FANAR_API_KEY"),
)
model_name = "Fanar"

def get_completion(messages):
    print(f"[DEBUG] Sending messages to Fanar API:\n{messages}\n---")
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
    )
    content = response.choices[0].message.content
    return content

def get_agent_response(user_input, conversation_history):
    """
    Gets a response from the AI agent, handling classification and tool use.
    Returns a tuple of (text_response, media_response, updated_history).
    `media_response` will be bytes if an image is generated, otherwise None.
    """
    conversation_history.append({"role": "user", "content": user_input})
    
    text_response = None
    media_response = None

    try:
        # 1. Classify the message to determine the correct system prompt
        classification_messages = [
            {"role": "system", "content": CLASSIFICATION_PROMPT}
        ] + conversation_history[-5:]
        
        classification_response = client.chat.completions.create(
            model=model_name,
            messages=classification_messages,
            temperature=0
        )
        category = classification_response.choices[0].message.content.strip().lower()
        if category not in SYSTEM_PROMPTS:
            category = 'general'
        print(f"[DEBUG] Message category: {category}")
        
        # 2. Handle the message using the full conversation history
        system_prompt = SYSTEM_PROMPTS[category]
        messages_to_send = [
            {"role": "system", "content": system_prompt}
        ] + conversation_history[-5:]
        
        print("[DEBUG] Sending messages to Fanar API:")
        print(messages_to_send)
        print("---")
        
        response = client.chat.completions.create(
            model=model_name,
            messages=messages_to_send,
            temperature=0
        )
        content = response.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": content})
        
        # 3. Agentic loop for tool calls
        while "TOOL_CALL:" in content:
            tool_call_match = re.search(r'TOOL_CALL:\s*({.*})', content, re.DOTALL)
            if not tool_call_match:
                print("[ERROR] Found 'TOOL_CALL:' but could not extract valid JSON.")
                break
            
            tool_call_str = tool_call_match.group(1)
            
            try:
                tool_call = json.loads(tool_call_str)
                tool = tool_call.get("tool")
                payload = tool_call.get("payload", {})
                
                if not tool:
                    print("[ERROR] Malformed tool call, missing 'tool' key.")
                    break

                # --- Handle Image Generation Tool ---
                if tool == "generate_image":
                    prompt = payload.get("prompt")
                    if not prompt:
                        text_response = "I need a prompt to generate an image. Please tell me what you want to create."
                        break
                    
                    image_bytes = generate_image_from_prompt(prompt)
                    if image_bytes:
                        media_response = image_bytes
                        text_response = "Here is the image you requested:"
                    else:
                        text_response = "Sorry, I was unable to generate the image."
                    break # End the loop after handling image generation
                    
                # Handle special date formatting for all calendar tools
                if tool.startswith("calendar_"):
                    for key in ["start", "end", "time_min", "time_max"]:
                        if key in payload:
                            time_str = payload.get(key)
                            if isinstance(time_str, str) and time_str.startswith(('today', 'tomorrow')):
                                try:
                                    dt_obj = parse_special_time_format(time_str)
                                    if dt_obj:
                                        utc_dt = dt_obj.astimezone(pytz.UTC)
                                        payload[key] = utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                                except ValueError as e:
                                    print(f"[ERROR] Could not parse date '{time_str}': {e}")
                                    
                if tool == "calendar_search_event":
                    user_prompt = ""
                    for i in range(len(conversation_history) - 2, -1, -1):
                        if conversation_history[i]['role'] == 'user':
                            user_prompt = conversation_history[i]['content'].lower()
                            break
                    
                    if "tomorrow" in user_prompt or "today" in user_prompt:
                        when = "tomorrow" if "tomorrow" in user_prompt else "today"
                        time_min, time_max = get_utc_iso_range(when)
                        payload.setdefault("time_min", time_min)
                        payload.setdefault("time_max", time_max)
                        payload.setdefault("query", "")
                
                mcp_client = MCPClient()
                print(f"[DEBUG] Calling MCP tool '{tool}' with payload:", payload)
                result = mcp_client.call(tool, payload)
                print("[Tool Result]", result)
                
                tool_result_message = {"role": "user", "content": f'The tool returned this result: {json.dumps(result)}.'}
                conversation_history.append(tool_result_message)
                
                messages_for_next_step = [
                    {"role": "system", "content": system_prompt}
                ] + conversation_history[-5:]
                
                print("[DEBUG] Sending messages to Fanar API for next step:")
                print(messages_for_next_step)
                print("---")
                
                next_response = client.chat.completions.create(
                    model=model_name,
                    messages=messages_for_next_step,
                    temperature=0
                )
                content = next_response.choices[0].message.content
                conversation_history.append({"role": "assistant", "content": content})
            
            except json.JSONDecodeError:
                content = "Error: Invalid JSON in tool call"
                break
            except Exception as e:
                content = f"Error calling tool: {str(e)}"
                break
        
        text_response = content if text_response is None else text_response
        return text_response, media_response, conversation_history[-5:]

    except Exception as e:
        print(f"Error: {str(e)}")
        return "Sorry, I encountered an error. Please try again.", None, conversation_history[-5:]

def generate_image_from_prompt(prompt: str):
    """
    Generates an image from a text prompt by calling the MCP server.
    Returns the image as bytes, or None if an error occurs.
    """
    try:
        print(f"[DEBUG] Calling MCP Server for ImageGen with prompt: '{prompt}'")
        mcp_client = MCPClient()
        payload = {"prompt": prompt}
        result = mcp_client.call("generate_image", payload)
        
        if result and "image_b64" in result:
            image_b64 = result["image_b64"]
            image_bytes = base64.b64decode(image_b64)
            return image_bytes
            
        print(f"[ERROR] MCP Server did not return image data. Response: {result}")
        return None
        
    except Exception as e:
        print(f"[ERROR] Unexpected error in generate_image_from_prompt: {e}")
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
        print(f"[ERROR] API Request Error: {e}")
        return f"Sorry, there was an error calling the vision API: {e}"
    except Exception as e:
        print(f"[ERROR] Unexpected error in get_vision_response: {e}")
        return "Sorry, an unexpected error occurred while processing the image."

def main():
    load_dotenv()
    
    print("Welcome to the Fanar LLM CLI (MCP tool mode). Type 'exit' to quit.")
    print("[DEBUG] Your system timezone is:", get_localzone())
    
    conversation_history = []
    
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() == 'exit':
            break
            
        final_response, updated_history = get_agent_response(user_input, conversation_history)
        conversation_history = updated_history
        print("Assistant:", final_response)

if __name__ == "__main__":
    main()
