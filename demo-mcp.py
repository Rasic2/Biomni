from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

# 初始化环境变量
load_dotenv(find_dotenv())

# Initialize the model
model = init_chat_model("azure_openai:gpt-4o")


async def make_dpa_graph():
    # Set up MCP client
    client = MultiServerMCPClient(
        {
            "dpa": {
                "url": "http://eyjo1360201.bohrium.tech:50001/sse",
                "transport": "sse",
            }
        }
    )
    tools = await client.get_tools()
    dpa_agent = create_react_agent(
        model=model,
        tools=tools,
        name="dpa_expert",
        prompt="You are a dpa expert. Always use one tool at a time."
    )
    return dpa_agent


# async with client.session("dpa") as session:
#     tools = await load_mcp_tools(session)
#
#
# async def get_tools():
#     return await client.get_tools()


# # Define call_model function
# async def call_model(state: MessagesState):
#     messages = state["messages"]
#     model_with_tools = model.bind_tools(await get_tools())
#     response = await model_with_tools.ainvoke(messages)
#     return {"messages": [response]}
#
#
# async def tool_node(state: MessagesState):
#     # return ToolNode(await get_tools())
#     return await get_tools()


#
#
# def should_continue(state: MessagesState):
#     messages = state["messages"]
#     last_message = messages[-1]
#     if last_message.tool_calls:
#         return "tools"
#     return END
#
#
# # Define the graph
# graph = (
#     StateGraph(MessagesState)
#     .add_node("call_model", call_model)
#     .add_node("tools", tool_node)
#     .add_edge(START, "call_model")
#     .add_conditional_edges("call_model", should_continue)
#     .add_edge("tools", "call_model")
#     .compile(name="New Graph")
# )

# dpa_agent = create_react_agent(
#     model=model,
#     tools=[tool_node],
#     name="dpa_expert",
#     prompt="You are a dpa expert. Always use one tool at a time."
# )

async def app():
    dpa_agent = await make_dpa_graph()

    # Create supervisor workflow
    workflow = create_supervisor(
        [dpa_agent],
        model=model,
        prompt=(
            "You are a team supervisor managing a dpa expert. "
            "For dpa problems, use dpa_agent."
        )
    )
    # Compile and run
    app = workflow.compile()

    return app
# result = app.invoke({
#     "messages": [
#         {
#             "role": "user",
#             "content": "what's the combined headcount of the FAANG companies in 2024?"
#         }
#     ]
# })
