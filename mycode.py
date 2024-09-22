from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

load_dotenv()


def tellJoke(topic: str):

    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o",  # or your deployment
        api_version="2023-03-15-preview",  # or your api version
        # other params...
    )

    prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")

    chain = prompt | llm | StrOutputParser()

    out = chain.invoke({"topic": topic})
    return out


if __name__ == "__main__":

    joke = tellJoke("bears")
    print(joke)
