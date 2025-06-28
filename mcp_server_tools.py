import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import base64
import serpapi

load_dotenv()
FANAR_API_KEY = os.getenv("FANAR_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

app = FastAPI(title="MCP Tools Server (No Email/Calendar)")

# Initialize Fanar OpenAI Client
fanar_client = OpenAI(
    base_url="https://api.fanar.qa/v1",
    api_key=FANAR_API_KEY,
)

class ImageGeneratePayload(BaseModel):
    prompt: str

class WebSearchPayload(BaseModel):
    query: str

@app.post("/mcp/generate_image")
def mcp_generate_image(payload: ImageGeneratePayload):
    """Generates an image from a prompt."""
    print(f"[DEBUG] Received image generation request: {payload.prompt}")
    try:
        print("[DEBUG] Calling Fanar image generation API...")
        response = fanar_client.images.generate(
            model="Fanar-ImageGen-1",
            prompt=payload.prompt,
            response_format="b64_json"
        )
        print("[DEBUG] Fanar API responded.")
        image_b64 = response.data[0].b64_json
        if not image_b64:
            print("[DEBUG] No image data returned from Fanar API.")
            raise HTTPException(status_code=500, detail="API returned no image data.")
        return {"result": "Image generated successfully", "image_b64": image_b64}
    except Exception as e:
        print(f"[ERROR] in image generation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate image: {str(e)}")

@app.post("/mcp/web_search")
def mcp_web_search(payload: WebSearchPayload):
    """Performs a web search using SerpApi and returns the results."""
    if not SERPAPI_API_KEY:
        raise HTTPException(status_code=500, detail="SERPAPI_API_KEY not set in environment.")
    try:
        client = serpapi.Client(api_key=SERPAPI_API_KEY)
        results = client.search(
            q=payload.query,
            engine="google",
            google_domain="google.com",
            gl="us",
            hl="en"
        )
        if "error" in results:
            raise HTTPException(status_code=500, detail=results["error"])
        organic_results = results.get("organic_results", [])
        formatted_results = []
        for result in organic_results[:5]:
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
    uvicorn.run("mcp_server_tools:app", host="0.0.0.0", port=8010, reload=True) 