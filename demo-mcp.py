from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

# 初始化环境变量
load_dotenv(find_dotenv())

# Initialize the model
model = init_chat_model("azure_openai:gpt-4o")

# Set up MCP client
client = MultiServerMCPClient(
    {
        "dpa": {
            "url": "http://eyjo1360201.bohrium.tech:50001/sse",
            "transport": "sse",
        }
    }
)


async def get_tools():
    return await client.get_tools()


# Define call_model function
async def call_model(state: MessagesState):
    messages = state["messages"]
    model_with_tools = model.bind_tools(await get_tools())
    response = await model_with_tools.ainvoke(messages)
    return {"messages": [response]}


async def tool_node(state: MessagesState):
    return ToolNode(await get_tools())


def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END


# Define the graph
graph = (
    StateGraph(MessagesState)
    .add_node("call_model", call_model)
    .add_node("tools", tool_node)
    .add_edge(START, "call_model")
    .add_conditional_edges("call_model", should_continue)
    .add_edge("tools", "call_model")
    .compile(name="New Graph")
)
