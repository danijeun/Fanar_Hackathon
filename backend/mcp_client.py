"""
Client for interacting with the MCP server.
"""

import os
import requests
import json
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')

# Define which tools are handled by which server
MAIN_SERVER_TOOLS = {
    "list_calendar_events",
    "create_calendar_event",
    "arabic_email_agent",
    "english_email_agent",
    "send_gmail",
}
TOOLS_SERVER_TOOLS = {
    "generate_image",
    "web_search",
}

class MCPClient:
    """Client for interacting with the MCP servers via HTTP POST."""
    def __init__(self, main_server_url=None, tools_server_url=None):
        self.main_server_url = main_server_url or os.getenv("MCP_SERVER_URL") or "http://127.0.0.1:8000"
        self.tools_server_url = tools_server_url or os.getenv("MCP_TOOLS_SERVER_URL") or "http://127.0.0.1:8010"
        if not self.main_server_url:
            raise ValueError("MCP_SERVER_URL not set in environment.")
        self.tools = {
            "list_calendar_events": {"description": "Lists events from Google Calendar.", "payload": {"start": "str (YYYY-MM-DD HH:MM)", "end": "str (YYYY-MM-DD HH:MM)", "max_results": "int (optional)"}},
            "create_calendar_event": {"description": "Creates an event in Google Calendar.", "payload": {"summary": "str", "start": "str (YYYY-MM-DD HH:MM)", "end": "str (YYYY-MM-DD HH:MM)"}},
            "arabic_email_agent": {"description": "Generates a professional Arabic email and sends it.", "payload": {"body": "str", "recipient_name": "str"}},
            "english_email_agent": {"description": "Generates a professional English email and sends it.", "payload": {"body": "str", "recipient_name": "str"}},
            "send_gmail": {"description": "Sends an email via Gmail.", "payload": {"recipient": "email", "subject": "str", "body": "str", "image_b64": "str (optional)"}},
            # The following are only available on the tools server:
            "generate_image": {"description": "Generates an image from a text prompt.", "payload": {"prompt": "str"}},
            "web_search": {"description": "Performs a web search using Google.", "payload": {"query": "str"}},
        }
        # Map tool names to the correct server (main or tools)
        self.tool_server_map = {
            'list_calendar_events': self.main_server_url,
            'create_calendar_event': self.main_server_url,
            'arabic_email_agent': self.main_server_url,
            'english_email_agent': self.main_server_url,
            'send_gmail': self.main_server_url,
            # Tools server:
            'generate_image': self.tools_server_url,
            'web_search': self.tools_server_url
        }

    def get_tools_json_for_prompt(self, tool_names):
        """Returns a JSON string of specified tools for the system prompt."""
        subset_tools = {name: self.tools[name] for name in tool_names if name in self.tools}
        return json.dumps(subset_tools, indent=2)

    def execute_tool(self, tool_name, payload):
        """Executes a tool by calling the appropriate MCP server."""
        if tool_name not in self.tools:
            return {"error": f"Tool '{tool_name}' not found."}
        headers = {"Content-Type": "application/json"}
        # Decide which server to use
        if tool_name in ("english_email_agent", "arabic_email_agent", "list_calendar_events", "create_calendar_event", "send_gmail"):
            url = f"{self.main_server_url}/mcp/{tool_name}"
        elif tool_name in ("generate_image", "web_search"):
            url = f"{self.tools_server_url}/mcp/{tool_name}"
        else:
            return {"error": f"Tool '{tool_name}' is not assigned to any server."}
        print(f"[DEBUG] MCPClient calling URL: {url} with payload: {payload}")
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
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