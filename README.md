# Python Project Structure for Playwright + LangGraph Web Automation

Modern web automation projects combining browser automation with AI agents require sophisticated architectural patterns to manage complexity while maintaining maintainability. Based on comprehensive research of industry standards, established frameworks, and successful open-source projects, here are detailed recommendations for structuring a Playwright + LangGraph web browsing agent system.

## Recommended project architecture for Playwright + LangGraph automation

The optimal structure leverages Python's **src layout** with clear separation between browser management, AI agent logic, state handling, and operational concerns. This architecture supports complex multi-agent workflows while maintaining code clarity and testability.

### Complete project structure template

```
my-web-agent/
├── pyproject.toml                    # Modern Python packaging configuration
├── README.md                         # Project overview and quick start
├── .env.example                     # Environment variable template
├── docker-compose.yml               # Development environment setup
├── Dockerfile                       # Production containerization
├── src/
│   └── web_agent/
│       ├── __init__.py
│       ├── main.py                  # Application entry point
│       ├── agents/                  # LangGraph agent definitions
│       │   ├── __init__.py
│       │   ├── base_agent.py       # Base agent functionality
│       │   ├── web_navigator.py    # Browser navigation agent
│       │   ├── data_extractor.py   # Content extraction agent
│       │   ├── task_planner.py     # Planning and coordination agent
│       │   └── workflow.py         # Multi-agent orchestration
│       ├── browser/                 # Playwright browser management
│       │   ├── __init__.py
│       │   ├── manager.py          # Browser lifecycle management
│       │   ├── context_factory.py  # Browser context creation
│       │   ├── page_handler.py     # Page interaction utilities
│       │   └── session_state.py    # Session persistence
│       ├── pages/                   # Page Object Model components
│       │   ├── __init__.py
│       │   ├── base_page.py        # Base page functionality
│       │   ├── components/         # Reusable page components
│       │   │   ├── __init__.py
│       │   │   ├── navigation.py
│       │   │   └── forms.py
│       │   └── specific_pages/     # Site-specific page objects
│       ├── state/                   # State management
│       │   ├── __init__.py
│       │   ├── schema.py           # State type definitions
│       │   ├── memory.py           # Agent memory patterns
│       │   └── persistence.py      # State storage
│       ├── tools/                   # LangGraph tools and functions
│       │   ├── __init__.py
│       │   ├── browser_tools.py    # Browser interaction tools
│       │   ├── extraction_tools.py # Data extraction tools
│       │   └── validation_tools.py # Content validation tools
│       ├── utils/                   # Common utilities
│       │   ├── __init__.py
│       │   ├── helpers.py          # General helper functions
│       │   ├── validators.py       # Input/output validation
│       │   └── error_handling.py   # Error handling patterns
│       └── config/                  # Configuration management
│           ├── __init__.py
│           ├── settings.py         # Application settings
│           ├── environments/       # Environment-specific configs
│           └── secrets.py          # Secrets management
├── tests/                          # Testing structure
│   ├── __init__.py
│   ├── conftest.py                # Pytest fixtures and configuration
│   ├── unit/                      # Unit tests
│   │   ├── test_agents/
│   │   ├── test_browser/
│   │   └── test_utils/
│   ├── integration/               # Integration tests
│   │   ├── test_workflows/
│   │   └── test_browser_integration/
│   ├── e2e/                      # End-to-end tests
│   │   └── test_complete_scenarios/
│   └── fixtures/                 # Test data and fixtures
├── docs/                         # Documentation
│   ├── ARCHITECTURE.md
│   ├── API_REFERENCE.md
│   ├── DEPLOYMENT.md
│   └── examples/
└── scripts/                      # Utility scripts
    ├── setup_dev.py
    └── run_tests.py
```

## Industry-standard Python packaging and dependency management

Modern Python projects should adopt **pyproject.toml** as the primary configuration file with the src layout for improved testing and packaging reliability. This approach prevents common import issues and ensures proper package isolation.

### Essential configuration pattern

```toml
[build-system]
requires = ["hatchling>=1.26"]
build-backend = "hatchling.build"

[project]
name = "web-agent"
version = "1.0.0"
description = "Playwright + LangGraph web automation agent"
requires-python = ">=3.9"
dependencies = [
    "playwright>=1.40.0",
    "langgraph>=0.2.0",
    "langchain>=0.2.0",
    "pydantic>=2.5.0",
    "asyncio-mqtt>=0.13.0",
    "structlog>=23.2.0",
    "click>=8.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-playwright>=0.4.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.5.0",
    "pre-commit>=3.4.0",
]
test = [
    "pytest-cov>=4.1.0",
    "factory-boy>=3.3.0",
    "responses>=0.23.0",
]
docs = [
    "sphinx>=7.2.0",
    "sphinx-rtd-theme>=1.3.0",
]

[project.scripts]
web-agent = "web_agent.main:cli"

[tool.pytest.ini_options]
addopts = "--cov=src --cov-report=html --cov-report=term-missing"
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.black]
line-length = 88
target-version = ['py39']

[tool.ruff]
line-length = 88
target-version = "py39"
select = ["E", "F", "UP", "B", "SIM", "I"]

[tool.mypy]
python_version = "3.9"
strict = true
```

**Key advantages of this approach:**
- **Dependency isolation** prevents accidental imports during development
- **Modern tooling compatibility** with Poetry, uv, and other package managers
- **Comprehensive dependency groups** separate development, testing, and documentation requirements
- **Type safety** with mypy integration and strict checking

## Playwright browser automation module organization

Browser management requires careful separation between browser lifecycle, page interactions, and session state. The most effective pattern separates these concerns into specialized modules that can be independently tested and maintained.

### Browser management architecture

```python
# browser/manager.py
class BrowserManager:
    """Central browser lifecycle management."""
    
    def __init__(self, config: BrowserConfig):
        self.config = config
        self.browsers: Dict[str, Browser] = {}
        self.contexts: Dict[str, BrowserContext] = {}
    
    async def create_browser(self, browser_type: str = "chromium") -> Browser:
        """Create and configure browser instance."""
        browser = await playwright[browser_type].launch(
            headless=self.config.headless,
            slow_mo=self.config.slow_mo,
            devtools=self.config.devtools
        )
        self.browsers[browser_type] = browser
        return browser
    
    async def create_context(
        self, 
        browser: Browser, 
        session_state: Optional[Dict] = None
    ) -> BrowserContext:
        """Create isolated browser context with state."""
        context = await browser.new_context(
            viewport=self.config.viewport,
            storage_state=session_state,
            permissions=self.config.permissions
        )
        return context

# browser/page_handler.py
class PageHandler:
    """High-level page interaction patterns."""
    
    def __init__(self, page: Page):
        self.page = page
        self.logger = structlog.get_logger(__name__)
    
    async def smart_wait(self, selector: str, timeout: int = 30000):
        """Intelligent waiting with multiple strategies."""
        try:
            await self.page.wait_for_selector(selector, timeout=timeout)
        except TimeoutError:
            await self.page.wait_for_load_state("networkidle")
            await self.page.wait_for_selector(selector, timeout=5000)
    
    async def extract_structured_data(self, schema: Dict) -> Dict:
        """Extract data based on structured schema."""
        return await self.page.evaluate(
            """(schema) => {
                const result = {};
                for (const [key, selector] of Object.entries(schema)) {
                    const element = document.querySelector(selector);
                    result[key] = element ? element.textContent.trim() : null;
                }
                return result;
            }""",
            schema
        )
```

### Page Object Model for maintainable automation

**Component-based page objects** provide the most maintainable approach for complex web applications, allowing reusable components across different pages while maintaining clear boundaries.

```python
# pages/base_page.py
class BasePage:
    """Foundation for all page objects."""
    
    def __init__(self, page: Page):
        self.page = page
        self.handler = PageHandler(page)
    
    async def navigate_to(self, url: str):
        """Navigate with error handling and validation."""
        await self.page.goto(url)
        await self.wait_for_page_load()
    
    async def wait_for_page_load(self):
        """Wait for page to be fully loaded."""
        await self.page.wait_for_load_state("networkidle")

# pages/components/navigation.py
class NavigationComponent:
    """Reusable navigation component."""
    
    def __init__(self, page: Page):
        self.page = page
        self.menu_selector = '[data-testid="main-nav"]'
    
    async def navigate_to_section(self, section: str):
        """Navigate to specific section."""
        await self.page.click(f'{self.menu_selector} >> text="{section}"')
```

## LangGraph agent project structure patterns

LangGraph projects benefit from **state-first architecture** where state schemas drive the agent design. This approach ensures consistent data flow and enables sophisticated multi-agent coordination patterns.

### Agent organization strategy

```python
# agents/base_agent.py
from langgraph.graph import StateGraph
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """Core state schema for web automation agents."""
    messages: Annotated[list, add_messages]
    current_url: str
    task_objective: str
    extracted_data: Dict[str, Any]
    browser_session: str
    error_count: int
    completed_actions: List[str]

class BaseWebAgent:
    """Base class for web automation agents."""
    
    def __init__(self, llm, browser_manager: BrowserManager):
        self.llm = llm
        self.browser_manager = browser_manager
        self.tools = self._create_tools()
    
    def _create_tools(self) -> List[Tool]:
        """Create agent-specific tools."""
        return [
            self._create_navigation_tool(),
            self._create_extraction_tool(),
            self._create_validation_tool()
        ]
    
    def create_graph(self) -> StateGraph:
        """Build the agent's decision graph."""
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("planner", self.planning_node)
        graph.add_node("executor", self.execution_node)
        graph.add_node("validator", self.validation_node)
        
        # Define edges and conditions
        graph.add_edge("planner", "executor")
        graph.add_conditional_edges(
            "executor",
            self.should_validate,
            {"validate": "validator", "complete": "__end__"}
        )
        
        return graph.compile()

# agents/workflow.py
class MultiAgentWorkflow:
    """Coordinate multiple specialized agents."""
    
    def __init__(self, config: WorkflowConfig):
        self.navigator = WebNavigatorAgent(config.llm, config.browser_manager)
        self.extractor = DataExtractionAgent(config.llm, config.browser_manager)
        self.planner = TaskPlannerAgent(config.llm)
    
    async def execute_complex_task(self, task: ComplexTask) -> TaskResult:
        """Execute multi-step task with agent coordination."""
        plan = await self.planner.create_execution_plan(task)
        
        for step in plan.steps:
            if step.type == "navigation":
                await self.navigator.execute(step)
            elif step.type == "extraction":
                result = await self.extractor.execute(step)
                plan.update_context(result)
        
        return plan.get_final_result()
```

### State management and memory patterns

**Persistent state management** enables agents to maintain context across sessions and learn from previous interactions. This is crucial for complex web automation tasks that span multiple pages and sessions.

```python
# state/memory.py
class AgentMemory:
    """Persistent memory for agent learning."""
    
    def __init__(self, storage_backend: StorageBackend):
        self.storage = storage_backend
        self.short_term_memory = {}
    
    async def store_interaction(self, interaction: InteractionRecord):
        """Store successful interaction patterns."""
        await self.storage.save(
            key=f"interaction_{interaction.site}_{interaction.task_type}",
            value=interaction.model_dump(),
            ttl=86400 * 30  # 30 days
        )
    
    async def retrieve_similar_interactions(
        self, 
        site: str, 
        task_type: str
    ) -> List[InteractionRecord]:
        """Retrieve similar past interactions for context."""
        pattern = f"interaction_{site}_{task_type}*"
        records = await self.storage.scan(pattern)
        return [InteractionRecord.model_validate(r) for r in records]

# state/persistence.py
class StateCheckpointer:
    """LangGraph-compatible state checkpointing."""
    
    async def save_checkpoint(
        self, 
        config: RunnableConfig, 
        checkpoint: Checkpoint
    ) -> None:
        """Save checkpoint to persistent storage."""
        await self.storage.save(
            key=f"checkpoint_{config['configurable']['thread_id']}",
            value=checkpoint.model_dump()
        )
```

## Module separation patterns for complex browser automation

**Domain-driven module separation** provides the clearest architecture for complex automation projects. This approach groups related functionality while maintaining loose coupling between domains.

### Recommended separation strategy

1. **Browser Domain** (`browser/`) - All browser management and low-level interactions
2. **Agent Domain** (`agents/`) - AI decision-making and planning logic  
3. **Page Domain** (`pages/`) - Page Object Model and site-specific knowledge
4. **State Domain** (`state/`) - State management and memory persistence
5. **Tools Domain** (`tools/`) - Reusable automation tools and utilities
6. **Config Domain** (`config/`) - Configuration and environment management

```python
# Clear interface boundaries between domains
class BrowserDomain:
    """Encapsulates all browser-related functionality."""
    
    def __init__(self):
        self.manager = BrowserManager()
        self.page_factory = PageFactory()
        self.session_manager = SessionManager()

class AgentDomain:
    """Encapsulates AI agent functionality."""
    
    def __init__(self, browser_domain: BrowserDomain):
        self.browser = browser_domain
        self.memory = AgentMemory()
        self.workflow = MultiAgentWorkflow()
```

## Configuration management for multi-component systems

**Layered configuration architecture** handles the complexity of managing browser settings, LLM configurations, environment variables, and runtime parameters across different deployment environments.

### Environment-aware configuration pattern

```python
# config/settings.py
from pydantic import BaseSettings
from typing import Optional, Dict, Any

class BrowserConfig(BaseSettings):
    """Browser-specific configuration."""
    headless: bool = True
    slow_mo: int = 0
    viewport_width: int = 1920
    viewport_height: int = 1080
    timeout: int = 30000
    
    class Config:
        env_prefix = "BROWSER_"

class LLMConfig(BaseSettings):
    """LLM configuration across providers."""
    provider: str = "openai"
    model: str = "gpt-4"
    temperature: float = 0.1
    max_tokens: int = 2000
    api_key: Optional[str] = None
    
    class Config:
        env_prefix = "LLM_"

class ApplicationConfig(BaseSettings):
    """Main application configuration."""
    environment: str = "development"
    log_level: str = "INFO"
    debug: bool = False
    
    browser: BrowserConfig = BrowserConfig()
    llm: LLMConfig = LLMConfig()
    
    @property
    def is_production(self) -> bool:
        return self.environment == "production"

# config/environments/production.py
class ProductionConfig(ApplicationConfig):
    """Production environment overrides."""
    environment = "production"
    debug = False
    log_level = "WARNING"
    
    class Config:
        env_file = ".env.production"
```

## Testing architecture for modular automation projects

**Pyramid testing strategy** ensures comprehensive coverage while maintaining fast feedback loops. Web automation projects require specialized testing patterns for browser interactions, async operations, and AI agent behavior.

### Comprehensive testing structure

```python
# tests/conftest.py
import pytest
import asyncio
from playwright.async_api import async_playwright

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def browser():
    """Session-scoped browser for integration tests."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        yield browser
        await browser.close()

@pytest.fixture
async def page(browser):
    """Page fixture with automatic cleanup."""
    context = await browser.new_context()
    page = await context.new_page()
    yield page
    await context.close()

# tests/unit/test_agents/test_web_navigator.py
class TestWebNavigatorAgent:
    """Unit tests for navigation agent."""
    
    async def test_navigation_planning(self):
        """Test navigation strategy generation."""
        agent = WebNavigatorAgent(mock_llm, mock_browser)
        plan = await agent.create_navigation_plan("Login to dashboard")
        
        assert len(plan.steps) > 0
        assert any(step.action == "navigate" for step in plan.steps)

# tests/integration/test_workflows/test_complete_automation.py
class TestCompleteWorkflow:
    """Integration tests for full automation workflows."""
    
    async def test_end_to_end_data_extraction(self, browser):
        """Test complete data extraction workflow."""
        workflow = MultiAgentWorkflow(test_config)
        result = await workflow.execute_task(
            ComplexTask(
                objective="Extract product information from catalog",
                target_url="https://example-shop.com/catalog"
            )
        )
        
        assert result.success
        assert len(result.extracted_data) > 0
```

## Logging and monitoring organization

**Structured logging with correlation tracking** enables effective debugging and monitoring of complex multi-agent workflows. This approach provides visibility into agent decision-making and browser automation execution.

### Production-ready logging architecture

```python
# utils/logging.py  
import structlog
from typing import Any, Dict
import json
import sys

def configure_logging(config: ApplicationConfig):
    """Configure structured logging."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.JSONRenderer() if config.is_production 
            else structlog.dev.ConsoleRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(config.log_level)
        ),
        logger_factory=structlog.WriteLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )

# In agent code - contextual logging
logger = structlog.get_logger(__name__)

async def execute_navigation(self, target_url: str):
    """Execute navigation with comprehensive logging."""
    logger.info(
        "Starting navigation",
        target_url=target_url,
        session_id=self.session_id,
        agent_type=self.__class__.__name__
    )
    
    try:
        await self.page.goto(target_url)
        logger.info("Navigation successful", current_url=self.page.url)
    except Exception as e:
        logger.error(
            "Navigation failed",
            error=str(e),
            target_url=target_url,
            exc_info=True
        )
        raise
```

### Monitoring and observability patterns

```python
# utils/monitoring.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
automation_requests = Counter('automation_requests_total', 'Total automation requests')
task_duration = Histogram('automation_task_duration_seconds', 'Task execution time')
active_browser_sessions = Gauge('active_browser_sessions', 'Active browser sessions')

class MonitoringMixin:
    """Mixin to add monitoring to agents."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = time.time()
    
    def record_task_completion(self, success: bool):
        """Record task metrics."""
        automation_requests.inc()
        duration = time.time() - self.start_time
        task_duration.observe(duration)
        
        logger.info(
            "Task completed",
            duration=duration,
            success=success,
            agent_type=self.__class__.__name__
        )
```

## Documentation structure for comprehensive projects

**Living documentation architecture** ensures documentation stays current with the rapidly evolving codebase while serving both developers and users effectively.

### Structured documentation approach

```markdown
docs/
├── README.md                   # Project overview and quick start
├── ARCHITECTURE.md            # System design and patterns
├── API_REFERENCE.md           # Generated API documentation  
├── DEPLOYMENT.md              # Production deployment guide
├── user_guides/
│   ├── getting_started.md     # User onboarding
│   ├── configuration.md       # Configuration options
│   ├── examples/              # Usage examples
│   └── troubleshooting.md     # Common issues
├── development/
│   ├── setup.md              # Development environment
│   ├── testing.md            # Testing procedures
│   ├── contributing.md       # Contribution guidelines
│   └── debugging.md          # Debugging workflows
└── technical/
    ├── agent_design.md       # Agent architecture
    ├── browser_integration.md # Browser automation patterns
    └── performance.md        # Performance optimization
```

## Key implementation recommendations

**Start with proven patterns** from successful open-source projects while adapting them to your specific requirements. The **browser-use** project demonstrates excellent simplicity in agent design, while **LiteWebAgent** shows sophisticated multi-agent coordination patterns.

### Essential architectural principles

1. **Modular design with clear boundaries** - Each module should have a single responsibility and well-defined interfaces
2. **Async-first architecture** - Embrace Python's asyncio for browser automation and AI operations
3. **Configuration-driven behavior** - Make the system adaptable through configuration rather than code changes
4. **Comprehensive error handling** - Include retry mechanisms, graceful degradation, and detailed error reporting
5. **Observable operations** - Include logging, metrics, and tracing from the beginning
6. **Test-driven development** - Build testing into the architecture, not as an afterthought

### Scaling considerations

For production systems, consider **horizontal scaling patterns** with stateless agents, external state storage, and proper resource management. The multi-agent pattern enables natural scaling by distributing different types of work across specialized agents.

This architectural approach provides a solid foundation for building maintainable, scalable Playwright + LangGraph web automation systems that can grow from prototype to production while maintaining code quality and operational reliability.