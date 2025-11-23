from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
load_dotenv()


def main():
    model = ChatOpenAI(model = "gpt-4o-mini", temperature=0)

    essayTemplate = """
        You are a helpful assistant that takes the name of famous person {person} and does 2 things.
        1. Writes a short biography for that person under 100 words.
        2. Give 2 amazing facts about that person.
    """

    prompt = PromptTemplate(input_variables=["person"], template=essayTemplate)

    chain = prompt | model
    response = chain.invoke([{"person": "Elon Musk"}, {"person": "Narendar Modi"}])
    print(response.content)


if __name__ == "__main__":
    main()
