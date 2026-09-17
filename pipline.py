from langgraph.graph import StateGraph, START, END

from state import NewsIntelState
from plan import plan_node
from search import search_node
from extract import extract_node, extract_profile_node
from call_model import get_langfuse_client, get_langfuse_handler


def build_graph():
    graph = StateGraph(NewsIntelState)

    graph.add_node("plan", plan_node)
    graph.add_node("search", search_node)
    graph.add_node("extract_news", extract_node)
    graph.add_node("extract_profile", extract_profile_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "search")

    graph.add_edge("search", "extract_news")
    graph.add_edge("search", "extract_profile")

    graph.add_edge("extract_news", END)
    graph.add_edge("extract_profile", END)

    return graph.compile()


compiled_graph = build_graph()


def run_pipeline(initial_state: dict) -> dict:

    langfuse = get_langfuse_client()
    handler = get_langfuse_handler()

    with langfuse.start_as_current_observation(
        as_type="span",
        name="news_intel_pipeline",
    ) as span:
        span.update(input=initial_state)

        result = compiled_graph.invoke(
            initial_state,
            config={"callbacks": [handler]},
        )

        span.update(
            output={
                "analyzed_news_count": len(result.get("analyzed_news", [])),
                "has_profile": result.get("company_profile") is not None,
            }
        )

    return result



