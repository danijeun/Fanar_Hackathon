"""
Client for interacting with the MCP server.
"""

import os
import requests
import json
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')

KNOWN_TOOLS = {
    "list_calendar_events",
    "create_calendar_event",
    "send_gmail",
    "translate_text",
    "generate_image",
    "format_professional_arabic_email",
    "web_search",
    "generate_image_and_send_email",
}

class MCPClient:
    """Client for interacting with an external MCP server via HTTP POST."""
    def __init__(self, server_url=None):
        self.server_url = server_url or os.getenv("MCP_SERVER_URL") or "http://127.0.0.1:8000"
        if not self.server_url:
            raise ValueError("MCP_SERVER_URL not set in environment.")
        self.tools = {
            "send_gmail": {"description": "Sends an email via Gmail.", "payload": {"recipient": "email", "subject": "str", "body": "str", "image_b64": "str (optional)"}},
            "list_calendar_events": {"description": "Lists events from Google Calendar.", "payload": {"when": "str ('today' or 'tomorrow')"}},
            "create_calendar_event": {"description": "Creates an event in Google Calendar.", "payload": {"summary": "str", "start": "str (YYYY-MM-DD HH:MM)", "end": "str (YYYY-MM-DD HH:MM)"}},
            "translate_text": {"description": "Translates text to a target language.", "payload": {"text": "str", "target_lang": "str (e.g., 'ar')"}},
            "generate_image": {"description": "Generates an image from a text prompt.", "payload": {"prompt": "str"}},
            "format_professional_arabic_email": {"description": "Formats text into a professional Arabic email.", "payload": {"body": "str", "recipient_name": "str (optional)"}},
            "web_search": {"description": "Performs a web search using Google.", "payload": {"query": "str"}},
            "generate_image_and_send_email": {
                "description": "Generates an image from a text prompt and sends it to an email address.",
                "payload": {
                    "prompt": "str",
                    "recipient": "email",
                    "subject": "str",
                    "body": "str"
                }
            }
        }

    def get_tools_json_for_prompt(self, tool_names):
        """Returns a JSON string of specified tools for the system prompt."""
        subset_tools = {name: self.tools[name] for name in tool_names if name in self.tools}
        return json.dumps(subset_tools, indent=2)

    def execute_tool(self, tool_name, payload):
        """Executes a tool by calling the MCP server."""
        if tool_name not in self.tools:
            return {"error": f"Tool '{tool_name}' not found."}
            
        headers = {"Content-Type": "application/json"}
        url = f"{self.server_url}/mcp/{tool_name}"
        print(f"[DEBUG] MCPClient calling URL: {url} with payload: {payload}")
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=20) # Increased timeout
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError as ce:
            print(f"[ERROR] Could not connect to MCP server at {url}: {ce}")
            return {"error": f"Connection error: {ce}"}
        except requests.exceptions.Timeout as te:
            print(f"[ERROR] Request to MCP server timed out: {te}")
            return {"error": f"Timeout error: {te}"}
        except requests.exceptions.HTTPError as he:
            print(f"[ERROR] HTTP error from MCP server: {he} - {getattr(response, 'text', '')}")
            return {"error": f"HTTP error: {he}", "raw_response": getattr(response, 'text', '')}
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
            return {"error": str(e)} 