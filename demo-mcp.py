from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph_supervisor import create_supervisor

# 初始化环境变量
load_dotenv(find_dotenv())

# Initialize the model
model = init_chat_model("azure_openai:gpt-4o")

# Initialize MCP client
mcp_server_toolset = MultiServerMCPClient(
    {
        "dpa": {
            "url": "http://eyjo1360201.bohrium.tech:50001/sse",
            "transport": "sse",
        },
        "smiles": {
            "url": "http://ctoe1357111.bohrium.tech:50002/sse",
            "transport": "sse"
        }
    }
)


async def get_tools(server_name):
    return await mcp_server_toolset.get_tools(server_name=server_name)


async def dpa_llm_node(state: MessagesState):
    messages = state["messages"]
    tools = await get_tools(server_name="dpa")
    model_with_tools = model.bind_tools(tools)
    response = await model_with_tools.ainvoke(messages)
    return {"messages": [response]}


async def dpa_tool_node(state: MessagesState):
    tools = await get_tools(server_name="dpa")
    return ToolNode(tools)


def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tool_node"
    return END


dpa_agent_graph = (
    StateGraph(MessagesState)
    .add_node("dpa_llm_node", dpa_llm_node)
    .add_node("dpa_tool_node", dpa_tool_node)
    .add_edge(START, "dpa_llm_node")
    .add_conditional_edges(
        "dpa_llm_node",
        should_continue,
        {
            "tool_node": "dpa_tool_node",
            END: END
        }
    )
    .add_edge("dpa_tool_node", "dpa_llm_node")
    .compile(name="dpa_agent")
)


async def smiles_llm_node(state: MessagesState):
    messages = state["messages"]
    tools = await get_tools(server_name="smiles")
    model_with_tools = model.bind_tools(tools)
    response = await model_with_tools.ainvoke(messages)
    return {"messages": [response]}


async def smiles_tool_node(state: MessagesState):
    tools = await get_tools(server_name="smiles")
    return ToolNode(tools)


smiles_agent_graph = (
    StateGraph(MessagesState)
    .add_node("smiles_llm_node", dpa_llm_node)
    .add_node("smiles_tool_node", smiles_tool_node)
    .add_edge(START, "smiles_llm_node")
    .add_conditional_edges(
        "smiles_llm_node",
        should_continue,
        {
            "tool_node": "smiles_tool_node",
            END: END
        }
    )
    .add_edge("smiles_tool_node", "smiles_llm_node")
    .compile(name="smiles_agent")
)

supervisor_agent_graph = create_supervisor(
    agents=[dpa_agent_graph, smiles_agent_graph],
    model=model,
).compile()

# Define the main app
app = (
    StateGraph(MessagesState)
    .add_node("superior_agent", supervisor_agent_graph)
    .add_edge(START, "superior_agent")
    .add_edge("superior_agent", END)
    .compile()
)
