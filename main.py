from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from react_agent_prompt import REACT_PROMPT
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

@tool
def get_text_length(text: str) -> int:
    """ 
        Returns the length of given text by characters.
        Args:
            text (str): The input text whose length is to be calculated.
        Returns:
            int: The length of the input text in characters.
    """
    text = text.strip("'\n").strip('"')  # Remove leading/trailing quotes if any
    return len(text)

if __name__ == "__main__":
    print("Welcome to ReAct Langchain Agent")
    # print(get_text_length("Hello, World!"))  # Example usage
    tools = [get_text_length]
    reAct_prompt = PromptTemplate.from_template(template=REACT_PROMPT).partial(
        tools=tools, tool_names=", ".join([tool.name for tool in tools])
    )
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0, stop=["\nObservation", "Observation"])

    agent = {"input" : lambda x:x["input"] } | reAct_prompt | model
    response = agent.invoke({"input": "What is the length of the text 'Dog'?"}, verbose=True)
    print(response)

