"""System prompt, tool-calling loop and chat memory.

Run:  python -m src.assistant
"""
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq

from src import config
from src.tools import TOOL_MAP, TOOLS

SYSTEM_PROMPT = """You are a helpful AI assistant for a company. Decide yourself which tool to use:
- company_policy_search: for company rules (returns, refunds, shipping, warranty).
- get_order_status: for order tracking. If the user has not given an order ID, ask for it first.
- web_search: for latest or real-time information from the internet.
- For greetings or general chat, answer directly without any tool.
Answer only from tool results when you use a tool, keep answers short and clear,
and say so if the tools did not return the answer."""

MAX_TOOL_ROUNDS = 5


def run_turn(llm, messages: list) -> str:
    """Call the model, run any requested tools, and return the final answer."""
    ai = llm.invoke(messages)
    messages.append(ai)

    for _ in range(MAX_TOOL_ROUNDS):
        if not ai.tool_calls:
            break
        for call in ai.tool_calls:
            print(f"  [tool] {call['name']}({call['args']})")
            tool = TOOL_MAP.get(call["name"])
            try:
                result = tool.invoke(call["args"]) if tool else f"Unknown tool {call['name']}"
            except Exception as exc:  # keep the chat alive if a tool fails
                result = f"Tool error: {exc}"
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        ai = llm.invoke(messages)
        messages.append(ai)

    return ai.content


def main() -> None:
    config.require_keys("GROQ_API_KEY", "GOOGLE_API_KEY", "PINECONE_API_KEY", "TAVILY_API_KEY")

    llm = ChatGroq(model=config.GROQ_MODEL, temperature=0, api_key=config.GROQ_API_KEY)
    llm = llm.bind_tools(TOOLS)

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    print("AI Company Assistant ready. Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        messages.append(HumanMessage(content=user_input))
        try:
            print(f"Assistant: {run_turn(llm, messages)}\n")
        except Exception as exc:
            print(f"Error: {exc}\n")
            messages.pop()  # drop the failed turn so memory stays valid


if __name__ == "__main__":
    main()
