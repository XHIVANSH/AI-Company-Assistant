"""The three tools the assistant can call."""
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from tavily import TavilyClient

from src import config

# Mock order data. Replace get_order_status's lookup with a call to your real API.
ORDERS = {
    "ORD1001": {"status": "Delivered", "item": "Wireless Mouse", "eta": "Delivered on 2 Nov 2026"},
    "ORD1002": {"status": "Shipped", "item": "Mechanical Keyboard", "eta": "Arriving in 2 days"},
    "ORD1003": {"status": "Processing", "item": "USB-C Hub", "eta": "Ships within 24 hours"},
}


@tool
def web_search(query: str) -> str:
    """Search the web for latest or real-time information (news, current events,
    recent facts). Do NOT use this for company policies or order status."""
    client = TavilyClient(api_key=config.TAVILY_API_KEY)
    response = client.search(query=query, max_results=4)
    results = response.get("results", [])
    if not results:
        return "No web results found."
    return "\n\n".join(
        f"{r['title']}\n{r['content']}\nSource: {r['url']}" for r in results
    )


@tool
def get_order_status(order_id: str) -> str:
    """Look up the status of a customer order by its order ID (e.g. ORD1002).
    If the user has not given an order ID, ask for it instead of calling this."""
    order = ORDERS.get(order_id.strip().upper())
    if not order:
        return f"No order found with ID {order_id}."
    return (
        f"Order {order_id.upper()}: {order['item']} - {order['status']}. {order['eta']}."
    )


@tool
def company_policy_search(query: str) -> str:
    """Search the company's policy documents (returns, refunds, shipping, warranty,
    and other company rules). Use this for any question about company policies."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.EMBED_MODEL, google_api_key=config.GOOGLE_API_KEY
    )
    store = PineconeVectorStore(index_name=config.PINECONE_INDEX, embedding=embeddings)
    docs = store.similarity_search(query, k=4)
    if not docs:
        return "No relevant policy text found."
    return "\n\n---\n\n".join(d.page_content for d in docs)


TOOLS = [web_search, get_order_status, company_policy_search]
TOOL_MAP = {t.name: t for t in TOOLS}
