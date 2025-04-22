from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
#from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool


load_dotenv() #load environment variables from .env file

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]
    #can make this as complex as you want
    #or even make it a dataclass if you want

#choose an llm model to use
llm = ChatOpenAI(model="gpt-4.1-nano")
#llm2 = ChatAnthropic(model="claude-3-5-sonnet-20241022")

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools.
            Wrap the output in this format and provide no other text\n{format_instructions}
            """
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"), #can pass in multiple prompt variables
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools, #tools are things that the LLM/agent can use that we can either write ourself
    #or we can bring in from langchain Community Hub
    prompt=prompt#,
#    verbose=True,
#    output_parser=parser,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True#,
#    return_only_outputs=True,
)

query = input("What's up? ")
raw_response = agent_executor.invoke({"query": query})
#print(raw_response)

try:
    structured_response = parser.parse(raw_response.get("output")[0]["text"])
    print(structured_response)
except Exception as e:
    print(f"Error parsing response: {e}", f"Raw response: {raw_response}")
    structured_response = None



#response = llm.invoke("What is the capital of France?")
#print(response)
