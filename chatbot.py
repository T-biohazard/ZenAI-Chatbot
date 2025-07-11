import asyncio
from crawl4ai import AsyncWebCrawler
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage 
from langchain_groq import ChatGroq
from langchain_core.tools import BaseTool
from typing_extensions import TypedDict
from typing import Annotated, List, Optional, Type
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
import os
import re
from pydantic import BaseModel, Field

# loading the API keys stored in my .env 
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# creating a crawling tool
class WebCrawlerInput(BaseModel):
    url: str = Field(description="Crawl URL")
    extract_question: Optional[str] = Field(
        default=None, 
        description="Optional question to guide content extraction"
    )

class WebCrawlerTool(BaseTool):
    name: str = "web_crawler"
    description: str = "Crawl and extract content from web pages. Useful for getting detailed content from specific URLs."
    args_schema: Type[BaseModel] = WebCrawlerInput

    def _run(self, url: str, extract_question: Optional[str] = None) -> str:
        """Synchronous wrapper for the async crawler"""
        try:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            return loop.run_until_complete(self._arun(url, extract_question))
        except Exception as e:
            return f"Error crawling {url}: {str(e)}"

    async def _arun(self, url: str, extract_question: Optional[str] = None) -> str:
        """Asynchronous crawling implementation"""
        try:
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=url)
                
                if result.success:
                    # Truncate if too long -> with token limit 
                    content = result.markdown[:3000]  
                    if len(result.markdown) > 3000:
                        content += "\n\n[Content truncated due to length...]"
                    return f"Content from {url}:\n\n{content}"
                else:
                    return f"Failed to crawl {url}: {result.error_message}"
                    
        except Exception as e:
            return f"Error crawling {url}: {str(e)}"

# used tools
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=500)
arxiv_tool = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=500)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

tavily_tool = TavilySearchResults()
crawler_tool = WebCrawlerTool()

# all in one
tools = [arxiv_tool, wiki_tool, tavily_tool, crawler_tool]

# LLM + tools binding
llm = ChatGroq(model="deepseek-r1-distill-llama-70b")
llm_with_tools = llm.bind_tools(tools=tools)

# State Schema _ tool calling llm 

class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

def tool_calling_llm(state: State):
    """
    Enhanced LLM node that better handles compound questions
    """
    messages = state["messages"]
    
    # It is a custom system message to help with compound questions, added for better clarification! 
    system_prompt = """You are a helpful research assistant. When users ask compound questions (multiple requests in one message), make sure to address ALL parts of their question. 

If a user asks you to both crawl a paper AND find recent news about a topic, you should:
1. First use the web_crawler tool to get the paper content
2. Then use the tavily_search tool to find recent news about the topic
3. Provide a comprehensive response covering both requests

Always be thorough and don't miss any part of the user's request."""
    
    if not messages or not hasattr(messages[0], 'content') or messages[0].content != system_prompt:
        messages = [HumanMessage(content=system_prompt)] + messages
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# LangGraph workflow 
builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", "tool_calling_llm")
graph = builder.compile()

def enhance_user_message(content: str) -> str:
    """
    Enhanced user message processing to handle compound questions
    """
    #arXiv ID
    if re.match(r"^\d{4}\.\d{4,5}$", content.strip()):
        return f"Please find information about arXiv paper {content.strip()}"
    
    #arXiv URL
    arxiv_match = re.search(r"https?://arxiv\.org/abs/(\d{4}\.\d{4,5})", content)
    if arxiv_match:
        arxiv_id = arxiv_match.group(1)
        # Check if further informations are asked
        if any(keyword in content.lower() for keyword in ['news', 'recent', 'related', 'latest', 'current']):
            return f"Please do two things: 1) Crawl and summarize the arXiv paper {arxiv_id} from the URL {content.split()[0]}, and 2) Search for recent news and related information about the topics discussed in this paper."
        else:
            return f"Please crawl and summarize the arXiv paper {arxiv_id} from the URL {content.split()[0]}"
    
    # compound questions
    if any(connector in content.lower() for connector in ['also', 'and', 'plus', 'additionally']) and \
       any(keyword in content.lower() for keyword in ['news', 'recent', 'related', 'latest', 'current']):
        return f"Please handle this compound request: {content}. Make sure to address all parts of the question."
    
    return content

def run_chatbot(message, history: list) -> str:
    """
    Main chatbot function that processes messages and returns responses
    """
    try:
        messages = []
        
        # history message 
        for msg in history:
            if isinstance(msg, list) and len(msg) == 2:
                # Gradio history 
                messages.append(HumanMessage(content=enhance_user_message(msg[0])))
                if msg[1]:  
                    messages.append(AIMessage(content=msg[1]))
            elif isinstance(msg, dict):
                # Dictionary
                if msg["role"] == "user":
                    enhanced_content = enhance_user_message(msg["content"])
                    messages.append(HumanMessage(content=enhanced_content))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
    
        enhanced_message = enhance_user_message(message)
        
        # Add extra instruction
        if any(keyword in message.lower() for keyword in ['also', 'and', 'plus', 'additionally']):
            enhanced_message += "\n\nIMPORTANT: Please make sure to address ALL parts of this request. Don't skip any part of the question."
        
        messages.append(HumanMessage(content=enhanced_message))
        
        state_input = {"messages": messages}
        response = graph.invoke(state_input)
        
        #Find the last AI message from the response for history
        response_messages = response.get("messages", [])
    
        for msg in reversed(response_messages):
            if hasattr(msg, 'content') and msg.content:
                if hasattr(msg, 'type') and msg.type == "ai":
                    return msg.content
                elif hasattr(msg, '__class__') and msg.__class__.__name__ == "AIMessage":
                    return msg.content
        
        return "I'm sorry, I couldn't generate a response. Please try again."
    
    except Exception as e:
        print(f"Error in run_chatbot: {str(e)}")
        return f"An error occurred: {str(e)}"

def run_chatbot_sequential(message, history: list) -> str:
    """
    Sequential processing approach for compound questions - more reliable for complex requests
    """
    try:
        messages = []
        for msg in history:
            if isinstance(msg, list) and len(msg) == 2:
                messages.append(HumanMessage(content=enhance_user_message(msg[0])))
                if msg[1]:
                    messages.append(AIMessage(content=msg[1]))
            elif isinstance(msg, dict):
                if msg["role"] == "user":
                    enhanced_content = enhance_user_message(msg["content"])
                    messages.append(HumanMessage(content=enhanced_content))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
        
        arxiv_match = re.search(r"https?://arxiv\.org/abs/(\d{4}\.\d{4,5})", message)
        is_compound = arxiv_match and any(keyword in message.lower() for keyword in ['news', 'recent', 'related', 'latest', 'current'])
        
        if is_compound:
            arxiv_id = arxiv_match.group(1)
            arxiv_url = arxiv_match.group(0)
            
            crawl_message = f"Please crawl and summarize the arXiv paper from this URL: {arxiv_url}"
            messages.append(HumanMessage(content=crawl_message))
            
            state_input = {"messages": messages}
            response1 = graph.invoke(state_input)
            
            paper_response = ""
            for msg in reversed(response1.get("messages", [])):
                if hasattr(msg, 'content') and msg.content and hasattr(msg, 'type') and msg.type == "ai":
                    paper_response = msg.content
                    break

            messages.append(AIMessage(content=paper_response))
            news_message = f"Now search for recent news and related information about the topics discussed in arXiv paper {arxiv_id}"
            messages.append(HumanMessage(content=news_message))
            
            state_input = {"messages": messages}
            response2 = graph.invoke(state_input)
            
            news_response = ""
            for msg in reversed(response2.get("messages", [])):
                if hasattr(msg, 'content') and msg.content and hasattr(msg, 'type') and msg.type == "ai":
                    news_response = msg.content
                    break
            final_response = f"## Paper Analysis\n\n{paper_response}\n\n## Recent News & Related Information\n\n{news_response}"
            return final_response
        
        else:
            return run_chatbot(message, history)
    
    except Exception as e:
        print(f"Error in run_chatbot_sequential: {str(e)}")
        return f"An error occurred: {str(e)}"
