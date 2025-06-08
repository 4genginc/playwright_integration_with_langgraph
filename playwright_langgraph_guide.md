# Playwright Integration with LangGraph: Creating Web-Browsing AI Agents

## Overview

Combining Playwright with LangGraph creates powerful AI agents capable of autonomous web browsing, form filling, data extraction, and complex multi-step web interactions. This integration leverages Playwright's browser automation capabilities with LangGraph's agent orchestration framework.

## Architecture Components

### Core Technologies
- **Playwright**: Browser automation library for web interactions
- **LangGraph**: Framework for building stateful, multi-actor applications with LLMs
- **LangChain**: Provides LLM integration and tooling
- **Browser Context**: Manages browser sessions and state

### Key Benefits
- **Stateful Navigation**: Maintain browser state across multiple interactions
- **Visual Understanding**: Process screenshots and visual elements
- **Error Recovery**: Handle page load failures and navigation issues
- **Parallel Processing**: Manage multiple browser instances simultaneously

## Implementation Guide

### 1. Environment Setup

```bash
# Install required packages
pip install playwright langgraph langchain langchain-openai
pip install langchain-community

# Install browser binaries
playwright install chromium
```

### 2. Basic Browser Tool Implementation

```python
from playwright.async_api import async_playwright
from langchain.tools import BaseTool
from langgraph import StateGraph, END
from typing import Dict, Any, Optional
import asyncio
from pydantic import BaseModel, Field

class BrowserState(BaseModel):
    url: str = ""
    page_content: str = ""
    screenshot: Optional[bytes] = None
    form_data: Dict[str, Any] = Field(default_factory=dict)
    navigation_history: list = Field(default_factory=list)
    current_task: str = ""
    error_message: str = ""

class PlaywrightBrowserTool(BaseTool):
    name = "browser_navigator"
    description = "Navigate to web pages and interact with elements"
    
    def __init__(self):
        super().__init__()
        self.playwright = None
        self.browser = None
        self.page = None
    
    async def setup_browser(self):
        """Initialize browser instance"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=False,  # Set to True for production
            args=['--no-sandbox', '--disable-dev-shm-usage']
        )
        context = await self.browser.new_context(
            viewport={'width': 1280, 'height': 720},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        self.page = await context.new_page()
    
    async def navigate_to_url(self, url: str) -> Dict[str, Any]:
        """Navigate to a specific URL"""
        try:
            await self.page.goto(url, wait_until='networkidle')
            content = await self.page.content()
            title = await self.page.title()
            
            return {
                "success": True,
                "url": url,
                "title": title,
                "content": content[:1000],  # Truncate for token limits
                "current_url": self.page.url
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": url
            }
    
    async def take_screenshot(self) -> bytes:
        """Capture current page screenshot"""
        return await self.page.screenshot(full_page=True)
    
    async def extract_elements(self, selector: str) -> list:
        """Extract elements matching CSS selector"""
        elements = await self.page.query_selector_all(selector)
        results = []
        
        for element in elements:
            text = await element.text_content()
            tag_name = await element.evaluate('el => el.tagName')
            attributes = await element.evaluate('''
                el => {
                    const attrs = {};
                    for (let attr of el.attributes) {
                        attrs[attr.name] = attr.value;
                    }
                    return attrs;
                }
            ''')
            
            results.append({
                "text": text,
                "tag": tag_name,
                "attributes": attributes
            })
        
        return results
    
    async def click_element(self, selector: str) -> Dict[str, Any]:
        """Click on an element"""
        try:
            await self.page.click(selector)
            await self.page.wait_for_load_state('networkidle')
            return {"success": True, "action": f"clicked {selector}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def fill_form(self, form_data: Dict[str, str]) -> Dict[str, Any]:
        """Fill form fields"""
        results = []
        
        for selector, value in form_data.items():
            try:
                await self.page.fill(selector, value)
                results.append({"field": selector, "success": True})
            except Exception as e:
                results.append({"field": selector, "success": False, "error": str(e)})
        
        return {"form_results": results}
    
    async def cleanup(self):
        """Clean up browser resources"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
```

### 3. LangGraph Agent Implementation

```python
from langgraph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

class WebBrowsingAgent:
    def __init__(self, llm_api_key: str):
        self.llm = ChatOpenAI(
            model="gpt-4-vision-preview",  # For screenshot analysis
            api_key=llm_api_key,
            temperature=0
        )
        self.browser_tool = PlaywrightBrowserTool()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the agent's decision graph"""
        graph = StateGraph(BrowserState)
        
        # Add nodes
        graph.add_node("initialize", self._initialize_browser)
        graph.add_node("navigate", self._navigate_to_page)
        graph.add_node("analyze_page", self._analyze_page_content)
        graph.add_node("interact", self._interact_with_page)
        graph.add_node("extract_data", self._extract_data)
        graph.add_node("complete_task", self._complete_task)
        graph.add_node("handle_error", self._handle_error)
        
        # Define edges
        graph.add_edge("initialize", "navigate")
        graph.add_conditional_edges(
            "navigate",
            self._should_continue_after_navigation,
            {
                "analyze": "analyze_page",
                "error": "handle_error"
            }
        )
        graph.add_conditional_edges(
            "analyze_page",
            self._determine_next_action,
            {
                "interact": "interact",
                "extract": "extract_data",
                "complete": "complete_task"
            }
        )
        graph.add_edge("interact", "analyze_page")
        graph.add_edge("extract_data", "complete_task")
        graph.add_edge("complete_task", END)
        graph.add_edge("handle_error", END)
        
        graph.set_entry_point("initialize")
        return graph.compile()
    
    async def _initialize_browser(self, state: BrowserState) -> BrowserState:
        """Initialize browser and set up context"""
        await self.browser_tool.setup_browser()
        state.navigation_history.append("Browser initialized")
        return state
    
    async def _navigate_to_page(self, state: BrowserState) -> BrowserState:
        """Navigate to the target URL"""
        result = await self.browser_tool.navigate_to_url(state.url)
        
        if result["success"]:
            state.page_content = result["content"]
            state.navigation_history.append(f"Navigated to {result['url']}")
        else:
            state.error_message = result["error"]
        
        return state
    
    async def _analyze_page_content(self, state: BrowserState) -> BrowserState:
        """Analyze current page content using LLM"""
        # Take screenshot for visual analysis
        screenshot = await self.browser_tool.take_screenshot()
        state.screenshot = screenshot
        
        # Get page structure
        links = await self.browser_tool.extract_elements("a")
        forms = await self.browser_tool.extract_elements("form")
        buttons = await self.browser_tool.extract_elements("button")
        
        # Analyze with LLM
        analysis_prompt = f"""
        Current task: {state.current_task}
        Page content preview: {state.page_content[:500]}
        
        Available elements:
        - Links: {len(links)} found
        - Forms: {len(forms)} found  
        - Buttons: {len(buttons)} found
        
        Based on the current task and page content, what should be the next action?
        Options: interact, extract, complete
        """
        
        messages = [
            SystemMessage(content="You are a web browsing assistant. Analyze the page and determine the next action."),
            HumanMessage(content=analysis_prompt)
        ]
        
        response = await self.llm.ainvoke(messages)
        state.navigation_history.append(f"Page analyzed: {response.content}")
        
        return state
    
    async def _interact_with_page(self, state: BrowserState) -> BrowserState:
        """Interact with page elements based on task"""
        # This would contain task-specific interaction logic
        # For example, filling forms, clicking buttons, etc.
        
        if state.form_data:
            result = await self.browser_tool.fill_form(state.form_data)
            state.navigation_history.append(f"Form interaction: {result}")
        
        return state
    
    async def _extract_data(self, state: BrowserState) -> BrowserState:
        """Extract relevant data from the page"""
        # Extract data based on task requirements
        # This could involve scraping specific elements, tables, etc.
        
        data_elements = await self.browser_tool.extract_elements("[data-testid], .data-item, table")
        state.navigation_history.append(f"Data extracted: {len(data_elements)} elements")
        
        return state
    
    async def _complete_task(self, state: BrowserState) -> BrowserState:
        """Mark task as complete and cleanup"""
        state.navigation_history.append("Task completed successfully")
        await self.browser_tool.cleanup()
        return state
    
    async def _handle_error(self, state: BrowserState) -> BrowserState:
        """Handle errors and attempt recovery"""
        state.navigation_history.append(f"Error handled: {state.error_message}")
        await self.browser_tool.cleanup()
        return state
    
    def _should_continue_after_navigation(self, state: BrowserState) -> str:
        """Determine if navigation was successful"""
        return "error" if state.error_message else "analyze"
    
    def _determine_next_action(self, state: BrowserState) -> str:
        """Determine next action based on current state"""
        # This would contain more sophisticated logic
        # For now, simplified decision making
        
        if "form" in state.page_content.lower() and state.form_data:
            return "interact"
        elif "table" in state.page_content.lower() or "data" in state.current_task.lower():
            return "extract"
        else:
            return "complete"
    
    async def execute_task(self, url: str, task: str, form_data: Dict = None) -> Dict[str, Any]:
        """Execute a web browsing task"""
        initial_state = BrowserState(
            url=url,
            current_task=task,
            form_data=form_data or {}
        )
        
        final_state = await self.graph.ainvoke(initial_state)
        
        return {
            "success": not final_state.error_message,
            "navigation_history": final_state.navigation_history,
            "error": final_state.error_message,
            "final_url": final_state.url
        }
```

### 4. Usage Examples

```python
# Example 1: Simple web scraping
async def scrape_product_data():
    agent = WebBrowsingAgent(llm_api_key="your-api-key")
    
    result = await agent.execute_task(
        url="https://example-ecommerce.com/products",
        task="Extract product names and prices from the product listing page"
    )
    
    print(f"Task completed: {result['success']}")
    print(f"Navigation history: {result['navigation_history']}")

# Example 2: Form submission
async def submit_contact_form():
    agent = WebBrowsingAgent(llm_api_key="your-api-key")
    
    form_data = {
        "#name": "John Doe",
        "#email": "john@example.com",
        "#message": "Hello from the AI agent!"
    }
    
    result = await agent.execute_task(
        url="https://example.com/contact",
        task="Fill and submit the contact form",
        form_data=form_data
    )
    
    return result

# Example 3: Multi-step navigation
async def research_competitor_pricing():
    agent = WebBrowsingAgent(llm_api_key="your-api-key")
    
    result = await agent.execute_task(
        url="https://competitor-site.com",
        task="Navigate to pricing page and extract all plan details and costs"
    )
    
    return result
```

## Advanced Features

### 1. Error Recovery and Retry Logic

```python
class RobustWebAgent(WebBrowsingAgent):
    def __init__(self, llm_api_key: str, max_retries: int = 3):
        super().__init__(llm_api_key)
        self.max_retries = max_retries
    
    async def _handle_error_with_retry(self, state: BrowserState) -> BrowserState:
        """Enhanced error handling with retry logic"""
        retry_count = getattr(state, 'retry_count', 0)
        
        if retry_count < self.max_retries:
            state.retry_count = retry_count + 1
            state.error_message = ""  # Clear error for retry
            
            # Implement different retry strategies
            if "timeout" in state.error_message.lower():
                await asyncio.sleep(2)  # Wait before retry
            elif "element not found" in state.error_message.lower():
                await self.browser_tool.page.wait_for_timeout(1000)
            
            return state
        else:
            await self.browser_tool.cleanup()
            return state
```

### 2. Parallel Browser Management

```python
class MultiPageAgent:
    def __init__(self, llm_api_key: str, max_concurrent: int = 5):
        self.llm_api_key = llm_api_key
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_multiple_urls(self, url_tasks: list) -> list:
        """Process multiple URLs concurrently"""
        async def process_single_task(url_task):
            async with self.semaphore:
                agent = WebBrowsingAgent(self.llm_api_key)
                return await agent.execute_task(
                    url=url_task["url"],
                    task=url_task["task"]
                )
        
        tasks = [process_single_task(url_task) for url_task in url_tasks]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
```

## Best Practices

### Performance Optimization
- Use headless mode for production deployments
- Implement connection pooling for multiple browser instances
- Cache page content to avoid redundant requests
- Use CSS selectors efficiently for element targeting

### Error Handling
- Implement comprehensive timeout handling
- Add retry logic for transient failures
- Validate page loads before proceeding with interactions
- Handle dynamic content loading with proper waits

### Security Considerations
- Validate URLs before navigation
- Implement rate limiting to avoid overwhelming target sites
- Use user-agent rotation for large-scale operations
- Respect robots.txt and site terms of service

### Monitoring and Logging
- Log all navigation steps and decisions
- Monitor browser resource usage
- Track success/failure rates
- Implement alerting for critical failures

## Conclusion

The integration of Playwright with LangGraph creates powerful web-browsing AI agents capable of complex, multi-step web interactions. This combination provides the automation capabilities of Playwright with the intelligent decision-making of LangGraph, enabling sophisticated web automation workflows that can adapt to different scenarios and handle errors gracefully.

The key to success lies in proper state management, robust error handling, and intelligent decision-making logic that can navigate the complexities of modern web applications.