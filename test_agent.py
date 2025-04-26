import pytest
from unittest.mock import patch
from app import query_analysis_node, web_search_node, web_scraper_node, content_synthesis_node

# Mock state for simplicity
default_state = {
    "messages": [],
    "query": "What is the future of AI in healthcare?",
    "query_analysis": None,
    "search_results": None,
    "scraped_content": None,
    "memory": {},
    "error": None
}

# --- Test 1: Normal flow works fine
def test_normal_flow():
    state = default_state.copy()
    state = query_analysis_node(state)
    assert state["query_analysis"] is not None, "Query analysis should return a result."

    state = web_search_node(state)
    assert state["search_results"] is not None, "Search results should not be empty."

    state = web_scraper_node(state)
    assert isinstance(state["scraped_content"], list), "Scraped content should be a list."

    state = content_synthesis_node(state)
    assert "final_answer" in state, "Final answer should be generated."

# --- Test 2: LLM returns empty response during query analysis
@patch('app.client.chat.completions.create')
def test_query_analysis_failure(mock_llm_response):
    mock_llm_response.return_value.choices = [{'message': {'content': ''}}]

    state = default_state.copy()
    state = query_analysis_node(state)
    assert state["error"] is not None, "Should catch error if LLM gives empty response."
    assert "query_analysis" not in state or state["query_analysis"] is None, "Query analysis should not be set."

# --- Test 3: No search results found
@patch('app.search')
def test_no_search_results(mock_search):
    mock_search.return_value = {"organic_results": []}

    state = default_state.copy()
    state["query_analysis"] = {"optimized_search_terms": ["future of AI in healthcare"]}
    state = web_search_node(state)

    assert state["error"] is not None, "Should catch error when no search results are found."
    assert state["search_results"] is None, "No search results should be present."

# --- Test 4: Web scraping fails (bad URL or timeout)
@patch('app.requests.get')
def test_web_scraping_failure(mock_get):
    mock_get.side_effect = Exception("Timeout error")

    state = default_state.copy()
    state["query_analysis"] = {"optimized_search_terms": ["future of AI in healthcare"]}
    state["search_results"] = [{"link": "http://fake-url.com"}]

    state = web_scraper_node(state)

    # Even if scraping fails, agent should not crash
    assert isinstance(state["scraped_content"], list), "Scraped content should still be a list (maybe empty)."
    assert state["error"] is not None, "Error should be caught during scraping."

