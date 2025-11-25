from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
# from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage
# from tavily import TavilyClient
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
load_dotenv()


# travily = TavilyClient()

# @tool
# def search(query: str) -> str:
#     """
#     Tool that perform a search over internet.
#     Args:
#         query (str): The search query.
#     Returns:
#         str: The search results.
#     """

#     print(f"agam_gupta - Searching for: {query}")
#     return travily.search(query = query)

class JobPosting(BaseModel):
    title: str = Field(..., description="The title of the job posting")
    company: str = Field(..., description="The company offering the job")
    salary: str = Field(..., description="The salary for the job")
    location: str = Field(..., description="The location of the job")
    experience: str = Field(..., description="Required experience for the job")
    link: str = Field(..., description="Link to the job posting")

class JobPostingsResponse(BaseModel):
    jobs: list[JobPosting] = Field(..., description="A list of job postings")
    size: int = Field(..., description="Number of job postings returned")
def main():
    """Main function to execute the Search Agent using LangChain."""
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [TavilySearch()]
    agent = create_agent(model=model, tools=tools, response_format=JobPostingsResponse)
    response = agent.invoke({"messages": [HumanMessage(content="list 3 jobs from linkedin for GenAI developer in Bangalore for 0-2 years experience") ]})
    print("Agent Response: ", response)
    
if __name__ == "__main__":
    main()
