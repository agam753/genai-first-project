from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.messages import HumanMessage, AIMessage, SystemMessage
load_dotenv()

@tool()
def is_temperature_feasible_for_henry(temperature: float) -> bool:
    """
        Determines if the given temperature is feasible for Henry.
        Args:
            temperature (float): The temperature in celcius to evaluate.
        Returns:
            bool: True if the temperature is feasible for Henry, False otherwise.
    """
    return 18.0 <= temperature <= 30.0


def main():
    print("LangChain Function calling Example")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [TavilySearch(), is_temperature_feasible_for_henry]

    agent = create_agent(llm, tools, system_prompt="You are a helpful assistant expert in anaylzing weather data")
    response = agent.invoke({"messages": [HumanMessage(content="Compare the weather in San Fransico and New Delhi today. Is the temperature in each location feasible for Henry?")]})
    print("Agent Response:", response)

if __name__ == "__main__":
    main()
