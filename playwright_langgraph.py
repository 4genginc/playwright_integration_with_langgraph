#!/usr/bin/env python3
"""
Playwright + LangGraph Web Browsing AI Agent
Complete executable implementation for autonomous web browsing
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Literal
from dataclasses import dataclass, field
from datetime import datetime
import os
from dotenv import load_dotenv
from pathlib import Path
import base64

# Core dependencies
from playwright.async_api import async_playwright, Browser, Page, Playwright
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------
# Load .env for secrets
# ----------------------------------------
load_dotenv(override=True)

# ============================================================================
# STATE DEFINITIONS
# ============================================================================

@dataclass
class BrowserState:
    """State object for the web browsing agent"""
    # Navigation state
    current_url: str = ""
    target_url: str = ""
    page_title: str = ""
    page_content: str = ""
    
    # Task management
    task_description: str = ""
    task_type: str = ""  # navigate, extract, interact, search
    current_step: str = "initialize"
    
    # Interaction data
    form_data: Dict[str, str] = field(default_factory=dict)
    click_targets: List[str] = field(default_factory=list)
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    
    # Agent memory
    navigation_history: List[str] = field(default_factory=list)
    screenshot_path: str = ""
    page_elements: List[Dict] = field(default_factory=list)
    
    # Error handling
    error_message: str = ""
    retry_count: int = 0
    max_retries: int = 3
    
    # Completion state
    task_completed: bool = False
    success: bool = False

# ============================================================================
# BROWSER AUTOMATION TOOLS
# ============================================================================

class PlaywrightManager:
    """Manages Playwright browser instances and operations"""
    
    def __init__(self, headless: bool = True, viewport_width: int = 1280, viewport_height: int = 720):
        self.headless = headless
        self.viewport = {"width": viewport_width, "height": viewport_height}
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        
    async def start(self):
        """Initialize browser instance"""
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=self.headless,
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-blink-features=AutomationControlled',
                    '--disable-extensions'
                ]
            )
            
            context = await self.browser.new_context(
                viewport=self.viewport,
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            
            self.page = await context.new_page()
            logger.info("Browser initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize browser: {e}")
            return False
    
    async def navigate(self, url: str, wait_until: str = 'networkidle') -> Dict[str, Any]:
        """Navigate to URL and return page info"""
        try:
            response = await self.page.goto(url, wait_until=wait_until, timeout=30000)
            
            # Get basic page info
            title = await self.page.title()
            content = await self.page.content()
            current_url = self.page.url
            
            return {
                "success": True,
                "url": current_url,
                "title": title,
                "content": content,
                "status_code": response.status if response else 200
            }
            
        except Exception as e:
            logger.error(f"Navigation failed for {url}: {e}")
            return {
                "success": False,
                "error": str(e),
                "url": url
            }
    
    async def take_screenshot(self, path: str = "screenshot.png") -> str:
        """Take screenshot and return path"""
        try:
            await self.page.screenshot(path=path, full_page=True)
            logger.info(f"Screenshot saved to {path}")
            return path
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return ""
    
    async def extract_elements(self, selector: str = None) -> List[Dict]:
        """Extract page elements for analysis"""
        try:
            elements = []
            
            # Default selectors for common elements
            selectors = [
                'a[href]',  # Links
                'button',   # Buttons
                'input',    # Form inputs
                'form',     # Forms
                'h1, h2, h3', # Headers
                '[data-testid]', # Test elements
                '.btn, .button', # Common button classes
            ] if not selector else [selector]
            
            for sel in selectors:
                try:
                    page_elements = await self.page.query_selector_all(sel)
                    for element in page_elements[:10]:  # Limit to prevent overflow
                        text = await element.text_content() or ""
                        tag_name = await element.evaluate('el => el.tagName.toLowerCase()')
                        
                        # Get relevant attributes
                        attrs = await element.evaluate('''
                            el => {
                                const result = {};
                                if (el.href) result.href = el.href;
                                if (el.id) result.id = el.id;
                                if (el.className) result.class = el.className;
                                if (el.type) result.type = el.type;
                                if (el.name) result.name = el.name;
                                if (el.value) result.value = el.value;
                                return result;
                            }
                        ''')
                        
                        elements.append({
                            "selector": sel,
                            "tag": tag_name,
                            "text": text.strip()[:100],  # Truncate long text
                            "attributes": attrs
                        })
                        
                except Exception as e:
                    logger.debug(f"Failed to extract {sel}: {e}")
                    continue
            
            return elements
            
        except Exception as e:
            logger.error(f"Element extraction failed: {e}")
            return []
    
    async def click_element(self, selector: str, wait_for_navigation: bool = True) -> Dict[str, Any]:
        """Click an element"""
        try:
            # Wait for element to be visible
            await self.page.wait_for_selector(selector, timeout=10000)
            
            if wait_for_navigation:
                async with self.page.expect_navigation(timeout=15000):
                    await self.page.click(selector)
            else:
                await self.page.click(selector)
                await self.page.wait_for_timeout(1000)  # Brief wait
            
            return {"success": True, "action": f"clicked {selector}"}
            
        except Exception as e:
            logger.error(f"Click failed for {selector}: {e}")
            return {"success": False, "error": str(e)}
    
    async def fill_form(self, form_data: Dict[str, str]) -> Dict[str, Any]:
        """Fill form fields"""
        results = []
        
        for selector, value in form_data.items():
            try:
                await self.page.wait_for_selector(selector, timeout=5000)
                await self.page.fill(selector, value)
                results.append({"field": selector, "value": value, "success": True})
                
            except Exception as e:
                logger.error(f"Failed to fill {selector}: {e}")
                results.append({"field": selector, "success": False, "error": str(e)})
        
        return {"form_results": results}
    
    async def search_text(self, text: str) -> List[Dict]:
        """Search for text on the page"""
        try:
            # Use JavaScript to find text
            results = await self.page.evaluate(f'''
                () => {{
                    const searchText = "{text}";
                    const walker = document.createTreeWalker(
                        document.body,
                        NodeFilter.SHOW_TEXT
                    );
                    
                    const results = [];
                    let node;
                    
                    while (node = walker.nextNode()) {{
                        if (node.textContent.toLowerCase().includes(searchText.toLowerCase())) {{
                            const element = node.parentElement;
                            results.push({{
                                text: node.textContent.trim(),
                                tagName: element.tagName.toLowerCase(),
                                className: element.className || "",
                                id: element.id || ""
                            }});
                        }}
                    }}
                    
                    return results.slice(0, 10); // Limit results
                }}
            ''')
            
            return results
            
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return []
    
    async def cleanup(self):
        """Clean up browser resources"""
        try:
            if self.page:
                await self.page.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            logger.info("Browser cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

# ============================================================================
# LANGGRAPH AGENT IMPLEMENTATION
# ============================================================================

class WebBrowsingAgent:
    """LangGraph-based web browsing agent"""
    
    def __init__(self, openai_api_key: str, headless: bool = True):
        self.llm = ChatOpenAI(
            model="gpt-4",
            api_key=openai_api_key,
            temperature=0.1,
            max_tokens=1500
        )
        
        self.browser = PlaywrightManager(headless=headless)
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """Build the agent's state graph"""
        
        def route_after_initialization(state: BrowserState) -> str:
            if state.error_message:
                return "handle_error"
            return "navigate"
        
        def route_after_navigation(state: BrowserState) -> str:
            if state.error_message:
                return "handle_error"
            return "analyze_page"
        
        def route_after_analysis(state: BrowserState) -> str:
            if state.error_message:
                return "handle_error"
            elif state.task_completed:
                return "complete_task"
            elif state.task_type == "extract":
                return "extract_data"
            elif state.task_type == "interact":
                return "interact_with_page"
            elif state.task_type == "search":
                return "search_content"
            else:
                return "extract_data"  # Default action
        
        def route_after_interaction(state: BrowserState) -> str:
            if state.error_message and state.retry_count >= state.max_retries:
                return "handle_error"
            elif state.task_completed:
                return "complete_task"
            else:
                return "analyze_page"
        
        # Create graph
        graph = StateGraph(BrowserState)
        
        # Add nodes
        graph.add_node("initialize_browser", self._initialize_browser)
        graph.add_node("navigate", self._navigate_to_page)
        graph.add_node("analyze_page", self._analyze_page)
        graph.add_node("extract_data", self._extract_data)
        graph.add_node("interact_with_page", self._interact_with_page)
        graph.add_node("search_content", self._search_content)
        graph.add_node("complete_task", self._complete_task)
        graph.add_node("handle_error", self._handle_error)
        
        # Set entry point
        graph.set_entry_point("initialize_browser")
        
        # Add edges
        graph.add_conditional_edges("initialize_browser", route_after_initialization)
        graph.add_conditional_edges("navigate", route_after_navigation)
        graph.add_conditional_edges("analyze_page", route_after_analysis)
        graph.add_conditional_edges("extract_data", route_after_interaction)
        graph.add_conditional_edges("interact_with_page", route_after_interaction)
        graph.add_conditional_edges("search_content", route_after_interaction)
        
        # Terminal edges
        graph.add_edge("complete_task", END)
        graph.add_edge("handle_error", END)
        
        return graph.compile()
    
    async def _initialize_browser(self, state: BrowserState) -> BrowserState:
        """Initialize browser session"""
        logger.info("Initializing browser...")
        
        success = await self.browser.start()
        if success:
            state.current_step = "browser_ready"
            state.navigation_history.append("Browser initialized successfully")
        else:
            state.error_message = "Failed to initialize browser"
            
        return state
    
    async def _navigate_to_page(self, state: BrowserState) -> BrowserState:
        """Navigate to target URL"""
        logger.info(f"Navigating to: {state.target_url}")
        
        result = await self.browser.navigate(state.target_url)
        
        if result["success"]:
            state.current_url = result["url"]
            state.page_title = result["title"]
            state.page_content = result["content"][:2000]  # Truncate for token limits
            state.current_step = "page_loaded"
            state.navigation_history.append(f"Successfully navigated to {result['url']}")
            
            # Take screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = f"screenshot_{timestamp}.png"
            state.screenshot_path = await self.browser.take_screenshot(screenshot_path)
            
        else:
            state.error_message = result["error"]
            state.retry_count += 1
            
        return state
    
    async def _analyze_page(self, state: BrowserState) -> BrowserState:
        """Analyze page content and determine next action"""
        logger.info("Analyzing page content...")
        
        try:
            # Extract page elements
            state.page_elements = await self.browser.extract_elements()
            
            # Prepare analysis prompt
            elements_summary = []
            for elem in state.page_elements[:15]:  # Limit for token management
                elements_summary.append(f"- {elem['tag']}: {elem['text'][:50]}...")
            
            analysis_prompt = f"""
            TASK: {state.task_description}
            TASK TYPE: {state.task_type}
            
            CURRENT PAGE:
            Title: {state.page_title}
            URL: {state.current_url}
            
            AVAILABLE ELEMENTS:
            {chr(10).join(elements_summary)}
            
            PAGE CONTENT PREVIEW:
            {state.page_content[:800]}
            
            Based on the task and available page elements, determine:
            1. Is the task complete? (yes/no)
            2. What should be the next action?
            3. Any specific elements to interact with?
            
            Respond in JSON format:
            {{
                "task_completed": true/false,
                "next_action": "extract/interact/search/complete",
                "reasoning": "explanation of decision",
                "target_elements": ["selector1", "selector2"],
                "confidence": 0.0-1.0
            }}
            """
            
            messages = [
                SystemMessage(content="You are a web browsing assistant. Analyze pages and make decisions based on the given task."),
                HumanMessage(content=analysis_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            try:
                # Parse LLM response
                decision = json.loads(response.content)
                
                state.task_completed = decision.get("task_completed", False)
                state.task_type = decision.get("next_action", "extract")
                state.click_targets = decision.get("target_elements", [])
                
                reasoning = decision.get("reasoning", "No reasoning provided")
                state.navigation_history.append(f"Analysis: {reasoning}")
                
                logger.info(f"Analysis complete. Next action: {state.task_type}")
                
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                content = response.content.lower()
                if "complete" in content or "finished" in content:
                    state.task_completed = True
                elif "click" in content or "button" in content:
                    state.task_type = "interact"
                else:
                    state.task_type = "extract"
                
                state.navigation_history.append(f"Analysis (fallback): {response.content[:100]}")
            
        except Exception as e:
            logger.error(f"Page analysis failed: {e}")
            state.error_message = f"Analysis failed: {str(e)}"
            
        return state
    
    async def _extract_data(self, state: BrowserState) -> BrowserState:
        """Extract data from the current page"""
        logger.info("Extracting data from page...")
        
        try:
            # Extract different types of content
            extracted = {
                "title": state.page_title,
                "url": state.current_url,
                "timestamp": datetime.now().isoformat(),
                "elements": []
            }
            
            # Extract specific data based on common patterns
            for element in state.page_elements:
                if element["text"] and len(element["text"].strip()) > 5:
                    extracted["elements"].append({
                        "type": element["tag"],
                        "text": element["text"],
                        "attributes": element.get("attributes", {})
                    })
            
            # Look for structured data (tables, lists, etc.)
            tables = await self.browser.extract_elements("table")
            if tables:
                extracted["tables"] = [{"text": t["text"]} for t in tables[:3]]
            
            lists = await self.browser.extract_elements("ul, ol")
            if lists:
                extracted["lists"] = [{"text": l["text"]} for l in lists[:3]]
            
            state.extracted_data = extracted
            state.task_completed = True
            state.success = True
            
            state.navigation_history.append(f"Data extracted: {len(extracted['elements'])} elements")
            logger.info(f"Extraction complete: {len(extracted['elements'])} elements found")
            
        except Exception as e:
            logger.error(f"Data extraction failed: {e}")
            state.error_message = f"Extraction failed: {str(e)}"
            
        return state
    
    async def _interact_with_page(self, state: BrowserState) -> BrowserState:
        """Interact with page elements"""
        logger.info("Interacting with page elements...")
        
        try:
            interaction_results = []
            
            # Fill forms if form data is provided
            if state.form_data:
                form_result = await self.browser.fill_form(state.form_data)
                interaction_results.append(f"Form filled: {form_result}")
            
            # Click target elements
            for target in state.click_targets:
                if target and isinstance(target, str):
                    # Try to find and click the element
                    click_result = await self.browser.click_element(target, wait_for_navigation=False)
                    interaction_results.append(f"Clicked {target}: {click_result}")
                    
                    # Wait a bit between clicks
                    await asyncio.sleep(1)
            
            # If no specific targets, try common interactive elements
            if not state.form_data and not state.click_targets:
                # Look for submit buttons, login buttons, etc.
                common_targets = [
                    'button[type="submit"]',
                    'input[type="submit"]',
                    '.submit-btn',
                    '.login-btn',
                    'button:contains("Search")'
                ]
                
                for target in common_targets:
                    try:
                        elements = await self.browser.extract_elements(target)
                        if elements:
                            click_result = await self.browser.click_element(target, wait_for_navigation=False)
                            interaction_results.append(f"Auto-clicked {target}: {click_result}")
                            break
                    except:
                        continue
            
            state.navigation_history.extend(interaction_results)
            
            # Re-analyze page after interaction
            await asyncio.sleep(2)  # Wait for page changes
            
        except Exception as e:
            logger.error(f"Page interaction failed: {e}")
            state.error_message = f"Interaction failed: {str(e)}"
            
        return state
    
    async def _search_content(self, state: BrowserState) -> BrowserState:
        """Search for specific content on the page"""
        logger.info("Searching page content...")
        
        try:
            # Extract search terms from task description
            search_terms = []
            task_words = state.task_description.lower().split()
            
            # Look for quoted terms or important keywords
            for word in task_words:
                if len(word) > 3 and word not in ['find', 'search', 'look', 'page', 'website']:
                    search_terms.append(word)
            
            search_results = {}
            for term in search_terms[:5]:  # Limit search terms
                results = await self.browser.search_text(term)
                if results:
                    search_results[term] = results
            
            state.extracted_data = {
                "search_results": search_results,
                "search_terms": search_terms,
                "timestamp": datetime.now().isoformat()
            }
            
            state.task_completed = True
            state.success = True
            
            state.navigation_history.append(f"Search completed for terms: {search_terms}")
            
        except Exception as e:
            logger.error(f"Content search failed: {e}")
            state.error_message = f"Search failed: {str(e)}"
            
        return state
    
    async def _complete_task(self, state: BrowserState) -> BrowserState:
        """Complete the task and cleanup"""
        logger.info("Completing task...")
        
        state.current_step = "completed"
        state.task_completed = True
        
        if not state.success:
            state.success = len(state.extracted_data) > 0 or len(state.navigation_history) > 1
        
        completion_summary = {
            "task": state.task_description,
            "success": state.success,
            "final_url": state.current_url,
            "steps_taken": len(state.navigation_history),
            "data_extracted": bool(state.extracted_data),
            "screenshot": state.screenshot_path
        }
        
        state.navigation_history.append(f"Task completed: {completion_summary}")
        
        # Cleanup browser
        await self.browser.cleanup()
        
        return state
    
    async def _handle_error(self, state: BrowserState) -> BrowserState:
        """Handle errors and attempt recovery"""
        logger.error(f"Handling error: {state.error_message}")
        
        state.current_step = "error"
        
        # Attempt retry if within limits
        if state.retry_count < state.max_retries and "navigation" in state.error_message.lower():
            state.retry_count += 1
            state.error_message = ""  # Clear error for retry
            state.navigation_history.append(f"Retrying... (attempt {state.retry_count})")
            
            # Wait before retry
            await asyncio.sleep(2)
            
        else:
            state.task_completed = True
            state.success = False
            state.navigation_history.append(f"Task failed after {state.retry_count} retries: {state.error_message}")
            
            # Cleanup browser
            await self.browser.cleanup()
        
        return state
    
    async def execute_task(self, url: str, task: str, task_type: str = "extract", form_data: Dict[str, str] = None) -> Dict[str, Any]:
        """Execute a web browsing task"""
        
        # Initialize state
        initial_state = BrowserState(
            target_url=url,
            task_description=task,
            task_type=task_type,
            form_data=form_data or {}
        )
        
        logger.info(f"Starting task: {task}")
        logger.info(f"Target URL: {url}")
        
        try:
            # Execute the graph
            final_state = await self.graph.ainvoke(initial_state)
            
            # Prepare result
            result = {
                "success": final_state.success,
                "task": task,
                "url": url,
                "final_url": final_state.current_url,
                "extracted_data": final_state.extracted_data,
                "navigation_history": final_state.navigation_history,
                "screenshot": final_state.screenshot_path,
                "error": final_state.error_message,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            await self.browser.cleanup()
            
            return {
                "success": False,
                "task": task,
                "url": url,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# ============================================================================
# USAGE EXAMPLES AND MAIN EXECUTION
# ============================================================================

async def example_web_scraping():
    """Example: Web scraping task"""
    
    # Set your OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    agent = WebBrowsingAgent(api_key, headless=False)  # Set headless=True for production
    
    try:
        # Example 1: Extract news headlines
        result1 = await agent.execute_task(
            url="https://news.ycombinator.com",
            task="Extract the top 10 news headlines and their links",
            task_type="extract"
        )
        
        print("\n=== NEWS EXTRACTION RESULT ===")
        print(f"Success: {result1['success']}")
        print(f"Data extracted: {len(result1.get('extracted_data', {}).get('elements', []))} elements")
        print(f"Navigation steps: {len(result1['navigation_history'])}")
        
        # Example 2: Search for specific content
        result2 = await agent.execute_task(
            url="https://example.com",
            task="Find contact information and email addresses on the page",
            task_type="search"
        )
        
        print("\n=== SEARCH RESULT ===")
        print(f"Search success: {result2['success']}")
        if result2.get('extracted_data'):
            search_results = result2['extracted_data'].get('search_results', {})
            print(f"Search terms found: {list(search_results.keys())}")
        
        return [result1, result2]
        
    except Exception as e:
        print(f"Example execution failed: {e}")
        return None

async def example_form_interaction():
    """Example: Form filling and interaction"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    agent = WebBrowsingAgent(api_key, headless=False)
    
    # Example form data
    form_data = {
        "#name": "Test User",
        "#email": "test@example.com",
        "#message": "This is a test message from the AI agent"
    }
    
    result = await agent.execute_task(
        url="https://httpbin.org/forms/post",  # Test form URL
        task="Fill out the contact form with provided information",
        task_type="interact",
        form_data=form_data
    )
    
    print("\n=== FORM INTERACTION RESULT ===")
    print(f"Success: {result['success']}")
    print(f"Form filled: {bool(form_data)}")
    print(f"Steps taken: {len(result['navigation_history'])}")
    
    return result

async def run_custom_task():
    """Run a custom task with user input"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    # Get user input
    print("\n=== CUSTOM WEB BROWSING TASK ===")
    url = input("Enter URL to visit: ").strip()
    task = input("Enter task description: ").strip()
    
    task_type = input("Enter task type (extract/interact/search) [default: extract]: ").strip() or "extract"
    
    if not url or not task:
        print("URL and task are required!")
        return
    
    agent = WebBrowsingAgent(api_key, headless=False)
    
    result = await agent.execute_task(
        url=url,
        task=task,
        task_type=task_type
    )
    
    print(f"\n=== CUSTOM TASK RESULT ===")
    print(f"Success: {result['success']}")
    print(f"Final URL: {result['final_url']}")
    print(f"Error: {result.get('error', 'None')}")
    print(f"Screenshot: {result.get('screenshot', 'None')}")
    
    if result['extracted_data']:
        print(f"Data extracted: {len(result['extracted_data'])} items")
    
    print("\nNavigation History:")
    for i, step in enumerate(result['navigation_history'], 1):
        print(f"{i}. {step}")
    
    return result

# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main execution function"""
    
    print("Playwright + LangGraph Web Browsing Agent")
    print("=" * 50)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Menu options
    print("\nSelect an option:")
    print("1. Run web scraping example")
    print("2. Run form interaction example")
    print("3. Run custom task")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    try:
        if choice == "1":
            print("\n🚀 Running web scraping examples...")
            results = await example_web_scraping()
            
            if results:
                print("\n✅ Web scraping completed successfully!")
                # Save results to file
                with open("scraping_results.json", "w") as f:
                    json.dump(results, f, indent=2)
                print("Results saved to scraping_results.json")
            
        elif choice == "2":
            print("\n🚀 Running form interaction example...")
            result = await example_form_interaction()
            
            if result and result['success']:
                print("\n✅ Form interaction completed successfully!")
                # Save result to file
                with open("form_interaction_result.json", "w") as f:
                    json.dump(result, f, indent=2)
                print("Result saved to form_interaction_result.json")
            
        elif choice == "3":
            print("\n🚀 Running custom task...")
            result = await run_custom_task()
            
            if result:
                # Save result to file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"custom_task_result_{timestamp}.json"
                with open(filename, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"Result saved to {filename}")
            
        elif choice == "4":
            print("👋 Goodbye!")
            return
            
        else:
            print("❌ Invalid choice. Please select 1-4.")
            
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        logger.error(f"Main execution error: {e}")

# ============================================================================
# ADDITIONAL UTILITY FUNCTIONS
# ============================================================================

class WebBrowsingToolkit:
    """Additional utility functions for web browsing tasks"""
    
    @staticmethod
    async def batch_process_urls(api_key: str, url_tasks: List[Dict[str, str]], max_concurrent: int = 3):
        """Process multiple URLs concurrently"""
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_url(url_task):
            async with semaphore:
                agent = WebBrowsingAgent(api_key, headless=True)
                try:
                    result = await agent.execute_task(
                        url=url_task["url"],
                        task=url_task["task"],
                        task_type=url_task.get("task_type", "extract")
                    )
                    return result
                except Exception as e:
                    logger.error(f"Failed to process {url_task['url']}: {e}")
                    return {
                        "success": False,
                        "url": url_task["url"],
                        "error": str(e)
                    }
        
        # Process all URLs
        tasks = [process_single_url(url_task) for url_task in url_tasks]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    @staticmethod
    def save_results_to_csv(results: List[Dict], filename: str = "web_browsing_results.csv"):
        """Save results to CSV file"""
        import csv
        
        if not results:
            return
        
        # Get all unique keys from results
        all_keys = set()
        for result in results:
            all_keys.update(result.keys())
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=sorted(all_keys))
            writer.writeheader()
            
            for result in results:
                # Convert complex data to strings
                row = {}
                for key, value in result.items():
                    if isinstance(value, (dict, list)):
                        row[key] = json.dumps(value)
                    else:
                        row[key] = str(value) if value is not None else ""
                writer.writerow(row)
        
        print(f"Results saved to {filename}")

# ============================================================================
# ADVANCED EXAMPLES
# ============================================================================

async def example_ecommerce_scraping():
    """Advanced example: E-commerce product scraping"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    print("\n🛒 E-commerce Scraping Example")
    print("=" * 40)
    
    # Example e-commerce URLs (replace with real ones)
    ecommerce_tasks = [
        {
            "url": "https://books.toscrape.com/",
            "task": "Extract book titles, prices, and ratings from the first page",
            "task_type": "extract"
        },
        {
            "url": "https://books.toscrape.com/catalogue/category/books/mystery_3/index.html",
            "task": "Extract mystery book information including titles and prices",
            "task_type": "extract"
        }
    ]
    
    # Process URLs concurrently
    results = await WebBrowsingToolkit.batch_process_urls(
        api_key=api_key,
        url_tasks=ecommerce_tasks,
        max_concurrent=2
    )
    
    print(f"\n✅ Processed {len(results)} e-commerce pages")
    
    # Save results
    WebBrowsingToolkit.save_results_to_csv(results, "ecommerce_scraping_results.csv")
    
    return results

async def example_social_media_monitoring():
    """Advanced example: Social media monitoring"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    print("\n📱 Social Media Monitoring Example")
    print("=" * 40)
    
    agent = WebBrowsingAgent(api_key, headless=True)
    
    # Monitor different social platforms (replace with real URLs)
    monitoring_tasks = [
        {
            "url": "https://news.ycombinator.com/",
            "task": "Monitor trending technology discussions and extract top 5 posts",
            "task_type": "extract"
        }
    ]
    
    results = []
    for task in monitoring_tasks:
        result = await agent.execute_task(
            url=task["url"],
            task=task["task"],
            task_type=task["task_type"]
        )
        results.append(result)
        
        # Add delay between requests to be respectful
        await asyncio.sleep(2)
    
    print(f"\n✅ Monitored {len(results)} social media sources")
    
    # Save results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"social_media_monitoring_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {filename}")
    
    return results

async def example_competitive_analysis():
    """Advanced example: Competitive analysis"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    print("\n🏢 Competitive Analysis Example")
    print("=" * 40)
    
    # Example competitor URLs
    competitor_tasks = [
        {
            "url": "https://example-competitor1.com/pricing",
            "task": "Extract pricing information and feature comparisons",
            "task_type": "extract"
        },
        {
            "url": "https://example-competitor2.com/about",
            "task": "Extract company information, team size, and key features",
            "task_type": "extract"
        }
    ]
    
    # Note: Using placeholder URLs - replace with actual competitor sites
    print("⚠️  Note: This example uses placeholder URLs.")
    print("Replace with actual competitor websites for real analysis.")
    
    return {"message": "Competitive analysis template ready"}

# ============================================================================
# CONFIGURATION AND SETUP
# ============================================================================

def setup_environment():
    """Setup environment and dependencies"""
    
    print("🔧 Setting up environment...")
    
    # Create necessary directories
    directories = ["screenshots", "results", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    # Setup logging
    log_file = f"logs/web_browsing_{datetime.now().strftime('%Y%m%d')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    print("✅ Environment setup complete")

def check_dependencies():
    """Check if all required dependencies are installed"""
    
    required_packages = [
        "playwright",
        "langgraph", 
        "langchain",
        "langchain_openai",
        "openai"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All dependencies are installed")
    return True

# ============================================================================
# ENHANCED MAIN EXECUTION
# ============================================================================

async def enhanced_main():
    """Enhanced main function with more options"""
    
    print("🚀 Playwright + LangGraph Web Browsing Agent")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: Please set OPENAI_API_KEY environment variable")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    while True:
        print("\n📋 Select an option:")
        print("1. 🔍 Web scraping example")
        print("2. 📝 Form interaction example") 
        print("3. 🎯 Custom task")
        print("4. 🛒 E-commerce scraping")
        print("5. 📱 Social media monitoring")
        print("6. 🏢 Competitive analysis")
        print("7. 🔧 Batch process URLs")
        print("8. 🚪 Exit")
        
        choice = input("\nEnter your choice (1-8): ").strip()
        
        try:
            if choice == "1":
                print("\n🚀 Running web scraping examples...")
                results = await example_web_scraping()
                
            elif choice == "2":
                print("\n🚀 Running form interaction example...")
                result = await example_form_interaction()
                
            elif choice == "3":
                print("\n🚀 Running custom task...")
                result = await run_custom_task()
                
            elif choice == "4":
                print("\n🚀 Running e-commerce scraping...")
                results = await example_ecommerce_scraping()
                
            elif choice == "5":
                print("\n🚀 Running social media monitoring...")
                results = await example_social_media_monitoring()
                
            elif choice == "6":
                print("\n🚀 Running competitive analysis...")
                results = await example_competitive_analysis()
                
            elif choice == "7":
                print("\n🚀 Running batch URL processing...")
                # Get URLs from user
                print("Enter URLs and tasks (press Enter twice to finish):")
                url_tasks = []
                while True:
                    url = input("URL: ").strip()
                    if not url:
                        break
                    task = input("Task: ").strip()
                    if not task:
                        break
                    url_tasks.append({"url": url, "task": task})
                
                if url_tasks:
                    api_key = os.getenv("OPENAI_API_KEY")
                    results = await WebBrowsingToolkit.batch_process_urls(api_key, url_tasks)
                    print(f"✅ Processed {len(results)} URLs")
                else:
                    print("No URLs provided")
                
            elif choice == "8":
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please select 1-8.")
                continue
                
            print("\n" + "="*50)
            
        except KeyboardInterrupt:
            print("\n🛑 Operation cancelled by user")
            break
        except Exception as e:
            print(f"❌ Error during execution: {e}")
            logger.error(f"Main execution error: {e}")

if __name__ == "__main__":
    # Run the enhanced main function
    asyncio.run(enhanced_main())