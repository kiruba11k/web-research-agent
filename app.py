from io import StringIO
import os
import time
import requests
import streamlit as st
from serpapi.google_search import GoogleSearch
from bs4 import BeautifulSoup
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableLambda
import re
from typing import Annotated, Optional, List
import json
from typing import TypedDict
from openai import OpenAI
import random
import asyncio
import aiohttp
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
import pandas as pd
import random
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse
from io import StringIO


# Load API Key from Streamlit Secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

SERPAPI_KEY = st.secrets["SERPAPI_KEY"]
os.environ["SERPAPI_KEY"] = SERPAPI_KEY
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# State definition
class AgentState(TypedDict):
    messages: Annotated[List[dict], add_messages]
    query_plan: Optional[str]
    search_results: Optional[List[dict]]
    scraped_content: Optional[List[dict]]
    final_summary: Optional[str]
    error: Optional[str]
    memory: Optional[dict]  # Added memory to store information

# Step 1: Query Analysis
def query_analysis_node(state: AgentState):
    print("🧠 Query analysis started...")
    try:
        messages = state["messages"]
        user_query = messages[-1].content if hasattr(messages[-1], "content") else messages[-1]

        prompt = f"""
        You are a query analysis expert. Given this research query:
        "{user_query}"
        - Identify the intent (e.g., factual, opinion, news, history)
        - Break it into subcomponents
        - Generate optimized Google search terms
        Return your answer in strict JSON format.
        """

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )

        raw_output = response.choices[0].message.content
        if not raw_output:
            raise ValueError("Empty response from LLM")

        # Sanitize the LLM output to ensure it's valid JSON
        json_str = re.search(r'{.*}', raw_output, re.DOTALL)
        if not json_str:
            raise ValueError("Could not extract JSON from response")

        query_plan = json.loads(json_str.group(0))

        # Store the query plan in memory
        state["memory"] = {"query": user_query, "query_plan": query_plan}

        state["query_plan"] = query_plan

    except json.JSONDecodeError as e:
        state["error"] = f"JSON parsing error: {str(e)}\nRaw output:\n{raw_output}"
    except Exception as e:
        state["error"] = f"Error in query analysis: {str(e)}"
    
    return state

# Step 2: Web Search via SerpAPI
def score_result(result, query_terms):
    """Basic relevance scoring based on query term frequency in title and snippet."""
    title = result.get("title", "").lower()
    snippet = result.get("snippet", "").lower()
    combined = title + " " + snippet
    return sum(term.lower() in combined for term in query_terms)

def web_search_node(state: AgentState):
    print("🔍 Web search starts")
    try:
        # Retrieve the query plan from memory
        query_plan = state["memory"].get("query_plan", {})
        query_terms = query_plan.get("optimized_search_terms", [])
        intent = query_plan.get("intent", "").lower()
        if not query_terms:
            raise ValueError("No search terms generated")

        search_results = []
        MAX_PAGES = 2

        for term in query_terms:
            modified_term = term
            params = {
                "api_key": SERPAPI_KEY,
                "engine": "google",
                "num": 10,
            }

            # Adjust query and parameters based on intent
            if "news" in intent:
                params["tbm"] = "n"  # Google News tab
            elif "opinion" in intent:
                modified_term += " review OR Reddit OR Quora"

            for page in range(MAX_PAGES):
                params.update({
                    "q": modified_term,
                    "start": page * 10,
                })

                search = GoogleSearch(params)
                results = search.get_dict()

                organic = results.get("organic_results", [])
                if not organic:
                    print(f"[⚠️ Warning] No results on page {page+1} for: '{modified_term}'")
                    break

                for result in organic:
                    search_results.append({
                        "title": result.get("title"),
                        "link": result.get("link"),
                        "snippet": result.get("snippet", ""),
                        "score": score_result(result, query_terms)
                    })

        if not search_results:
            raise ValueError("No search results found for any query.")

        # Sort and filter top N based on score
        search_results = sorted(search_results, key=lambda r: r["score"], reverse=True)
        for r in search_results:
            r.pop("score")  # Optional: Remove internal score from final output

        state["search_results"] = search_results
        print("✅ Final ranked search results:", search_results[:5])

    except Exception as e:
        state["error"] = f"❌ Error in web search: {str(e)}"
    return state

# Step 3: Web Scraper
def is_scraping_allowed(url):
    try:
        domain = urlparse(url).scheme + "://" + urlparse(url).netloc
        rp = RobotFileParser()
        rp.set_url(f"{domain}/robots.txt")
        rp.read()
        return rp.can_fetch("*", url)
    except:
        return False  # When in doubt, don't scrape

def web_scraper_node(state: AgentState):
    print("🕷️ Web scraping started...")
    try:
        extracted_content = []
        seen_urls = set()

        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(max_retries=2)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/89.0',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Safari/537.36 Edge/91.0.864.59'
        ]
        headers = {'User-Agent': random.choice(user_agents)}

        for result in state.get("search_results", []):
            url = result.get("link")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            if not is_scraping_allowed(url):
                print(f"[🚫 Blocked] Scraping disallowed by robots.txt: {url}")
                continue

            try:
                response = session.get(url, headers=headers, timeout=8)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, 'html.parser')
                content = ""

                # Prefer semantic containers
                candidates = soup.find_all(['article', 'main', 'section', 'div'])
                if not candidates:
                    candidates = soup.find_all(['p', 'li'])

                content += " ".join(t.get_text(separator=" ", strip=True) for t in candidates)

                # Extract tables using pandas
                tables = soup.find_all('table')
                for table in tables:
                    try:
                        df = pd.read_html(str(table))[0]
                        content += "\n" + df.to_string(index=False)
                    except Exception:
                        continue

                content = re.sub(r'\s+', ' ', content).strip()

                if len(content) > 150:
                    extracted_content.append({
                        "url": url,
                        "content": content[:2000]
                    })

            except requests.exceptions.RequestException as e:
                print(f"[⚠️ Error] Failed to scrape {url} - {str(e)}")
                continue
            except Exception as e:
                print(f"[⚠️ General Error] {url}: {str(e)}")
                continue

        if not extracted_content:
            state["error"] = "No usable content could be extracted from the search results."
        else:
            state["scraped_content"] = extracted_content

    except Exception as e:
        state["error"] = f"Fatal scraping error: {str(e)}"

    print("✅ Scraping completed.")
    return state

# Step 4: Content Analyzer & Synthesizer
def content_synthesis_node(state: AgentState):
    print("🧠 Starting content synthesis...")

    try:
        messages = state.get("messages", [])
        if not messages or not hasattr(messages[-1], "content"):
            raise ValueError("User query is missing or improperly formatted.")

        query = messages[-1].content
        sources = state.get("scraped_content", [])

        if not sources:
            raise ValueError("No scraped content available for synthesis.")

        # Combine and limit text to avoid token overflow
        combined_chunks = []
        total_length = 0
        max_total_chars = 12000

        for src in sources:
            chunk = src.get("content", "")
            if not chunk:
                continue

            if total_length + len(chunk) > max_total_chars:
                break

            combined_chunks.append(f"[{src.get('url')}] {chunk.strip()}")
            total_length += len(chunk)

        if not combined_chunks:
            raise ValueError("Scraped content was empty or invalid.")

        combined_text = "\n\n".join(combined_chunks)

        prompt = f"""
You are an expert researcher.

User Query:
"{query}"

Below is content from top relevant web sources:
{combined_text}

Instructions:
- Summarize the key points across all sources
- Resolve any contradictions if found
- Present a coherent, well-structured answer
- Use bullet points or sections if helpful
- Reference the source URLs inline (e.g., [source])

Respond with a synthesized, human-readable summary:
"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )

        content = response.choices[0].message.content.strip()

        if not content:
            raise ValueError("Synthesis response was empty.")

        state["final_summary"] = content

    except Exception as e:
        state["error"] = f"Error during synthesis: {str(e)}"

    print("✅ Content synthesis completed.")
    return state


# Final Node
def final_output_node(state: AgentState):
    print("final process")
    if "error" in state:
        st.error(f"\nError occurred: {state['error']}")
    else:
        st.subheader("Final Research Report:")
        st.write(state.get("final_summary", "No summary generated."))
    return state

# LangGraph Definition
graph = StateGraph(AgentState)
graph.add_node("QueryAnalysis", query_analysis_node)
graph.add_node("WebSearch", web_search_node)
graph.add_node("Scraper",web_scraper_node)
graph.add_node("Synthesizer", content_synthesis_node)
graph.add_node("Final", final_output_node)

# Edges
graph.add_edge("__start__","QueryAnalysis")
graph.add_edge("QueryAnalysis", "WebSearch")
graph.add_edge("WebSearch", "Scraper")
graph.add_edge("Scraper", "Synthesizer")
graph.add_edge("Synthesizer", "Final")
graph.add_edge("Final","__end__")

# Compile the graph
app = graph.compile()

# Streamlit UI

# Streamlit UI
def main():
    # Custom CSS for styling
    st.markdown("""
        <style>
        .main-title {
            font-size: 36px;
            font-weight: bold;
            color: #4CAF50;
        }
        .sub-header {
            font-size: 18px;
            font-style: italic;
            color: #007BFF;
        }
        .button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            font-size: 16px;
        }
        .card {
            background-color: #F4F4F4;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        </style>
        """, unsafe_allow_html=True)

    # Title and description
    st.markdown('<h1 class="main-title">Web Research Agent</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enter a research query, and let the AI-powered agent conduct a thorough search and analysis.</p>', unsafe_allow_html=True)

    # User input
    query = st.text_input("Enter your research query:", "")
    st.markdown("### Helpful Tips:")
    st.markdown("- Make sure your query is clear and concise.")
    st.markdown("- You can ask for any topic, from science to history!")
    
    if st.button("Start Research", key="start_research", help="Click here to start the research process.", disabled=not query):
        with st.spinner("Processing..."):
            app.invoke({"messages": [{"role": "user", "content": query}]})

    elif not query:
        st.warning("Please enter a query to start the research.")

    # Loading indicator
    st.progress(0)

if __name__ == "__main__":
    main()
