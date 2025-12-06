from langchain_core.documents import Document
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai.chat_models import ChatOpenAI
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_core.tools import tool
from dotenv import load_dotenv
load_dotenv()



pdf_data = PyPDFLoader("./example_data/Research Paper.pdf").load()
print(f"Loaded {len(pdf_data)} pages from the PDF file.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
splitted_pdf_data = text_splitter.split_documents(pdf_data)
embedding_agent = OpenAIEmbeddings(model = "text-embedding-3-small")
vector_store = InMemoryVectorStore(embedding_agent)
vector_store.add_documents(splitted_pdf_data)


@tool(response_format="content_and_artifact")
def context_retriever(query: str, top_k: int = 3) -> tuple[str, list[Document]]:
    """
        A function for retrieving context based on a query.
        Args:
            query (str): The input query string.
            top_k (int): The number of top similar documents to retrieve.
        Returns:
            tuple[str, list[Document]]: serialized string and list of retrieved documents.
    """
    retrieved_response = vector_store.similarity_search(query, k=top_k)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_response
    )

    return serialized, retrieved_response

if __name__ == "__main__":
    print("Welcome to RAG Agent Demo!")
    doc = Document(
        page_content="This is a sample document for the RAG agent demo.",
        metadata={"source": "demo_source"}
    )

    print(doc.page_content)
    print(doc.metadata)

    # file_path = "./example_data/Sales_Sample_data.csv"
    # csv_data = CSVLoader(file_path).load()
    # print(f"Loaded {len(csv_data)} records from the CSV file.")
    # # print(csv_data[0].metadata)  # Print the first record as a sample
    # print(csv_data[0].page_content)  # Print the content of the first record as a sample
    # print("Second Line of CSV data: ")
    # print(csv_data[1].page_content)  # Print the metadata of the first record as a sample
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=1000, chunk_overlap=200, add_start_index=True
    # )
    # splitted_csv_data = text_splitter.split_documents(csv_data)
    # print(f"Split into {len(splitted_csv_data)} chunks.")

    # for c in splitted_csv_data[:10]:
    #     print(c.page_content)
    #     print(c.metadata)

    
    # print(f"Split into {len(splitted_pdf_data)} chunks.")

    # idx = 0
    # for c in splitted_pdf_data[:10]:
    #     print("Chunk Index:", idx)
    #     print(c.metadata)
    #     print(c.page_content)
    #     idx += 1

    
    # vector_1 = embedding_agent.embed_query(splitted_pdf_data[0].page_content)
    # print("Sample embedding vector for the first chunk has length  :", len(vector_1))
    # print(vector_1[:10])

    # serlialzed_string, response = context_retriever("What is the main contribution of the research paper?", top_k=3)
    # print(f"Retrieved {len(response)} similar documents:")
    # print(serlialzed_string)
    # for doc in response:
    #     print(doc.metadata)
    #     print(doc.page_content)

    prompt = (
        "You have access to a tool that retrieves context from a blog post. "
        "Use the tool to help answer user queries."
    )
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_agent(model, tools=[context_retriever])

    agent_response = agent.invoke({"messages": [HumanMessage(content="What is Online Kabadiwala. Summarize in 3 bullet points")]})
    print("Agent Response: ", agent_response)

    messages = agent_response["messages"] or []

    for msg in messages:
        print("MessageId is ", msg.id)
        print(f"{msg.content}")

