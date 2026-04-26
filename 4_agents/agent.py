from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
import datetime

load_dotenv()

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """Returns the current date and time in the specified format."""
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

llm = ChatOpenAI(model="gpt-4")

system_prompt = "You are a helpful assistant. Use the provided tools to answer questions."

agent = create_agent(
    model=llm,
    tools=[get_system_time],
    system_prompt=system_prompt
)

query = (
    "What is the current time in London? (You are in India). "
    "Just show the current time and not the date."
)

# Agents now work directly with a list of messages
result = agent.invoke({"messages": [("user", query)]})

# The final assistant response is the last message
print(result["messages"][-1].content)