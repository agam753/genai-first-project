from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from output_format import AgentResponse
from react_agent_prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from langchain_tavily import TavilySearch
from langchain_core.prompts import PromptTemplate


tools = [TavilySearch()]

react_system_prompt = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "tools", "tool_names", "format_instructions", "agent_scratchpad"],
).partial(
    tools=tools,
    tool_names=", ".join([tool.name for tool in tools]),
    format_instructions=AgentResponse.model_json_schema(),
)

def main():
    print("Hello from reAct_search_agent!")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    


if __name__ == "__main__":
    main()