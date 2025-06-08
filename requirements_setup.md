# Playwright + LangGraph Web Browsing Agent Setup

## Requirements

### System Requirements
- Python 3.8 or higher
- Internet connection
- OpenAI API key

### Python Dependencies

Create a `requirements.txt` file with the following content:

```txt
playwright==1.40.0
langgraph==0.0.62
langchain==0.1.0
langchain-openai==0.0.2
langchain-community==0.0.10
openai==1.6.1
pydantic==2.5.0
asyncio
```

## Installation Steps

### 1. Clone or Download the Code
```bash
# Save the main Python code as web_browsing_agent.py
```

### 2. Install Python Dependencies
```bash
# Create virtual environment (recommended)
python -m venv web_agent_env
source web_agent_env/bin/activate  # On Windows: web_agent_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Install Playwright Browsers
```bash
# Install browser binaries
playwright install chromium

# If you encounter permission issues on Linux:
sudo playwright install-deps chromium
```

### 4. Set OpenAI API Key
```bash
# Set environment variable (replace with your actual API key)
export OPENAI_API_KEY="sk-your-openai-api-key-here"

# On Windows:
set OPENAI_API_KEY=sk-your-openai-api-key-here
```

### 5. Run the Agent
```bash
python web_browsing_agent.py
```

## Quick Start Examples

### Example 1: Simple Web Scraping
```python
import asyncio
import os
from web_browsing_agent import WebBrowsingAgent

async def quick_test():
    agent = WebBrowsingAgent(os.getenv("OPENAI_API_KEY"), headless=False)
    
    result = await agent.execute_task(
        url="https://news.ycombinator.com",
        task="Extract the top 5 news headlines",
        task_type="extract"
    )
    
    print(f"Success: {result['success']}")
    print(f"Headlines found: {len(result.get('extracted_data', {}).get('elements', []))}")

# Run the test
asyncio.run(quick_test())
```

### Example 2: Form Interaction
```python
async def form_test():
    agent = WebBrowsingAgent(os.getenv("OPENAI_API_KEY"), headless=False)
    
    form_data = {
        "#email": "test@example.com",
        "#message": "Hello from AI agent!"
    }
    
    result = await agent.execute_task(
        url="https://httpbin.org/forms/post",
        task="Fill out the contact form",
        task_type="interact",
        form_data=form_data
    )
    
    return result

asyncio.run(form_test())
```

## Configuration Options

### Browser Settings
```python
# Headless mode for production
agent = WebBrowsingAgent(api_key, headless=True)

# Custom viewport size
browser = PlaywrightManager(headless=False, viewport_width=1920, viewport_height=1080)
```

### Task Types
- `"extract"` - Extract data from pages
- `"interact"` - Fill forms and click elements  
- `"search"` - Search for specific content
- `"navigate"` - Multi-step navigation

### Error Handling
```python
# Custom retry settings
state = BrowserState()
state.max_retries = 5  # Increase retry attempts
```

## Troubleshooting

### Common Issues

1. **"Browser failed to launch"**
   ```bash
   # Install browser dependencies
   playwright install-deps
   ```

2. **"OpenAI API Error"**
   ```bash
   # Check API key is set correctly
   echo $OPENAI_API_KEY
   ```

3. **"Permission denied" on Linux**
   ```bash
   # Install system dependencies
   sudo apt-get update
   sudo apt-get install -y chromium-browser
   ```

4. **"Module not found" errors**
   ```bash
   # Reinstall dependencies
   pip install --upgrade -r requirements.txt
   ```

### Performance Tips

1. **Use headless mode for production**
   ```python
   agent = WebBrowsingAgent(api_key, headless=True)
   ```

2. **Batch process multiple URLs**
   ```python
   results = await WebBrowsingToolkit.batch_process_urls(
       api_key, url_tasks, max_concurrent=3
   )
   ```

3. **Adjust timeouts for slow sites**
   ```python
   # In PlaywrightManager.navigate()
   response = await self.page.goto(url, timeout=60000)  # 60 seconds
   ```

## File Structure

After setup, your project should look like:
```
web-browsing-agent/
├── web_browsing_agent.py      # Main agent code
├── requirements.txt           # Dependencies
├── screenshots/              # Generated screenshots
├── results/                  # Output files
├── logs/                     # Log files
└── README.md                # This file
```

## Usage Examples

The agent provides several built-in examples:

1. **Web Scraping** - Extract data from web pages
2. **Form Interaction** - Fill and submit forms
3. **Custom Tasks** - Define your own browsing tasks
4. **E-commerce Scraping** - Product information extraction
5. **Social Media Monitoring** - Track posts and trends
6. **Competitive Analysis** - Monitor competitor websites
7. **Batch Processing** - Handle multiple URLs at once

## Safety and Ethics

- Always respect website terms of service
- Implement rate limiting to avoid overwhelming servers
- Use appropriate delays between requests
- Respect robots.txt files
- Consider the legal implications of web scraping
- Don't scrape personal or sensitive information without permission

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the code comments and documentation
3. Test with simple examples first
4. Ensure all dependencies are properly installed

## License

This code is provided as an educational example. Please ensure compliance with all applicable laws and website terms of service when using this tool.