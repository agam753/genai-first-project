from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, ToolMessage
from dotenv import load_dotenv
load_dotenv()



@tool()
def get_text_length(text: str) -> int:
    """
        This function returns the length of the input text.
        :param text: Input text string
        :return: Length of the text
    """
    return len(text)

def find_tool_by_name(tools, name: str):
    for tool in tools:
        if tool.name == name:
            return tool
    return ValueError(f"Tool with name {name} not found.")

if __name__ == "__main__":
    print("Agent using LangChain's .bind_tools() method")

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [get_text_length]
    model_with_tools = model.bind_tools(tools)
    
    messages = [HumanMessage(content="What is the length of the text 'Dog'?")]
    while True:
        ai_message = model_with_tools.invoke(messages)
        

        tool_calls = getattr(ai_message, "tool_calls", None) or []
        if len(tool_calls) > 0:
            messages.append(ai_message)
            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool_id = tool_call.get("id")
                tool_args = tool_call.get("args", {})

                tool_to_call = find_tool_by_name(tools, tool_name)

                tool_response = tool_to_call.invoke(tool_args)
                print(f"Tool {tool_name} called with args {tool_args}, got response: {tool_response}")

                messages.append(ToolMessage(content=str(tool_response), tool_call_id=tool_id))
        else:
            print("Final Response from Agent:", ai_message.content)
            break

