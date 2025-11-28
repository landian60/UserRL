import http.client
import json
import time
import os
from pathlib import Path

def get_dashscope_api_key():
    """Get DashScope API key from secret.json or environment variable."""
    # Try to load from secret.json in project root
    try:
        secret_path = Path(__file__).parent.parent / "secret.json"
        if secret_path.exists():
            with open(secret_path) as f:
                secret_data = json.load(f)
                # Try dashscope_key first, then dashscope_api_key, then serper_key (for backward compatibility)
                return secret_data.get("dashscope_key") or secret_data.get("dashscope_api_key") or secret_data.get("serper_key", "")
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        pass
    
    # Try to load from current directory secret.json
    try:
        with open("secret.json") as f:
            secret_data = json.load(f)
            return secret_data.get("dashscope_key") or secret_data.get("dashscope_api_key") or secret_data.get("serper_key", "")
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        pass
    
    # Try environment variable
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("SERPER_API_KEY")
    if api_key:
        return api_key
    
    raise ValueError("DashScope API key not found. Please provide it in secret.json (dashscope_key or dashscope_api_key) or DASHSCOPE_API_KEY environment variable.")

_api_key = None

def _get_api_key():
    """Lazy load API key to avoid errors at import time."""
    global _api_key
    if _api_key is None:
        _api_key = get_dashscope_api_key()
    return _api_key

def search_serper(query, num=10):
    """Search using DashScope search augmentation API."""
    api_key = _get_api_key()
    
    # Handle mock API key for testing
    if api_key == "mock_api_key_for_testing":
        return f"Mock search results for '{query}':\n1. Mock Result 1\n- Snippet: This is a mock search result for testing purposes.\n2. Mock Result 2\n- Snippet: Another mock result to demonstrate the search functionality."
    
    # Use DashScope search augmentation API
    conn = http.client.HTTPSConnection("dashscope.aliyuncs.com")
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    # DashScope search augmentation generation API
    # Using qwen-search model for search-enhanced generation
    payload = json.dumps({
        "model": "qwen-search",
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ]
        },
        "parameters": {
            "result_format": "message"
        }
    })

    try_time = 0
    max_retries = 5
    data = None
    
    while try_time < max_retries:
        try:
            conn.request("POST", "/api/v1/services/aigc/search-generation/generation", payload, headers)
            res = conn.getresponse()
            response_data = res.read()
            response_data = response_data.decode("utf-8")
            data = json.loads(response_data)
            
            # Check if request was successful
            if res.status == 200:
                if data.get("output") or data.get("message"):
                    break
                elif data.get("code"):
                    error_msg = data.get("message", f"API error: {data.get('code')}")
                    if try_time >= max_retries - 1:
                        return f"Search Error: {error_msg}"
            else:
                if try_time >= max_retries - 1:
                    return f"Search Error: API returned status {res.status}, response: {response_data[:200]}"
            
        except json.JSONDecodeError as e:
            if try_time >= max_retries - 1:
                error = f"Search Error: Invalid JSON response - {e}"
                print(error)
                return error
        except Exception as e:
            if try_time >= max_retries - 1:
                error = f"Search Error: {e}"
                print(error)
                return error
        
        try_time += 1
        time.sleep(2)
        # Create new connection for retry
        conn = http.client.HTTPSConnection("dashscope.aliyuncs.com")
    
    if try_time >= max_retries or data is None:
        return "Search Error: Timeout or no response"
    
    try:
        output = ""
        index = 1
        
        # Parse DashScope search results
        # Try different response formats
        result_data = data.get("output", {})
        if not result_data:
            result_data = data
        
        # Check for message format (new API format)
        if "message" in result_data:
            message = result_data["message"]
            if isinstance(message, dict):
                content = message.get("content", "")
                if content:
                    output += f"{str(index)}. Search Result\n- Answer: {content}\n"
                    index += 1
        
        # Check for text format
        if "text" in result_data and not output:
            text = result_data["text"]
            if text:
                output += f"{str(index)}. Search Result\n- Answer: {text}\n"
                index += 1
        
        # Check for search results in various formats
        search_results = result_data.get("search_results", [])
        if not search_results:
            search_results = result_data.get("results", [])
        if not search_results:
            search_results = result_data.get("references", [])
        
        for item in search_results[:num]:
            if isinstance(item, dict):
                title = item.get("title", item.get("name", item.get("url", "Untitled")))
                snippet = item.get("snippet", item.get("content", item.get("description", item.get("text", ""))))
                
                if title or snippet:
                    if not title:
                        title = "Search Result"
                    if snippet:
                        output += f"{str(index)}. {title}\n- Snippet: {snippet}\n"
                        index += 1
                        if index > num:
                            break
        
        # If no structured results, return the text/message content
        if not output:
            if "text" in result_data:
                return result_data["text"]
            elif "message" in result_data:
                message = result_data["message"]
                if isinstance(message, dict):
                    return message.get("content", "No search results found.")
                return str(message)
            else:
                return "No search results found."
        
        return output.strip()
    
    except Exception as e:
        error = f"Search Error: {e}"
        print(error)
        print(f"Response data: {json.dumps(data, indent=2)[:500]}")
        return error

if __name__ == "__main__":
    query = "How's the weather in Beijing"
    print(search_serper(query, num=3))
