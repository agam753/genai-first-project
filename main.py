from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage, SystemMessage
load_dotenv()

def main():
    print("Hello from gen-ai-first-project!")

    model = ChatOpenAI(model = "gpt-4o", temperature=0)
    
    agent = create_agent(
        model=model,
        tools=[],
        )

    response = model.invoke("What's the capital of France?")
    print(response.content)

    conversation = [
        SystemMessage("You are a helpful assistant that translates English to French."),
        HumanMessage("Translate: I love programming."),
        AIMessage("J'adore la programmation."),
        HumanMessage("Translate: I love building applications.")
    ]

    response = model.invoke(conversation)
    print(response.content)

if __name__ == "__main__":
    main()
