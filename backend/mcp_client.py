"""
Client for interacting with the MCP server.
"""

import os
import requests

class MCPClient:
    """Client for interacting with an external MCP server via HTTP POST."""
    def __init__(self, server_url=None):
        self.server_url = server_url or os.getenv("MCP_SERVER_URL") or "http://127.0.0.1:8000"
        if not self.server_url:
            raise ValueError("MCP_SERVER_URL not set in environment.")

    def call(self, tool, payload):
        headers = {"Content-Type": "application/json"}
        url = f"{self.server_url}/mcp/{tool}"
        print(f"[DEBUG] MCPClient calling URL: {url}")
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
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