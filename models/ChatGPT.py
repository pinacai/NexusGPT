import os
from models.__init__ import prepareDataset
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic.v1.error_wrappers import ValidationError

# Loading Local API Keys
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# output parser
output_parser = StrOutputParser()

# Datasets
TCC_Dataset = prepareDataset("dataset/TCC_Dataset.csv")
CAIT_Dataset = prepareDataset("dataset/CAIT_Dataset.csv")


class ChatGPT:
    def __init__(self) -> None:
        self.taskClassificationPrompt = ChatPromptTemplate.from_messages(TCC_Dataset)
        self.generalAssistantPrompt = ChatPromptTemplate.from_messages(CAIT_Dataset)

    def chainInitializer(self, llm):
        self.taskClassificationChain = (
            self.taskClassificationPrompt | llm | output_parser
        )
        self.generalAssistantChain = self.generalAssistantPrompt | llm | output_parser

    # For classifying the user query into specific task category
    def classifyTaskCategory(self, user_input):
        try:
            aiResponse = self.taskClassificationChain.invoke({"text": user_input})
            response = {"error_occurred": False, "category": aiResponse, "error": None}
        except Exception as e:
            response = {"error_occurred": True, "category": None, "error": str(e)}
        return response

    # General & specialized task assistance
    def generalAssistant(self, user_input, chatHistory):
        try:
            # Extend the chat prompt with the previous chat history
            self.generalAssistantPrompt.extend(chatHistory)
            # Append the current query to the chat prompt
            self.generalAssistantPrompt.append(user_input)
            aiResponse = self.generalAssistantChain.invoke({"text": user_input})
            response = {"error_occurred": False, "response": aiResponse, "error": None}

        except Exception as e:
            response = {"error_occurred": True, "response": None, "error": str(e)}

        return response


class ChatGPT_3_5:
    def __init__(self):
        try:
            self.llm = ChatOpenAI(
                openai_api_key=OPENAI_API_KEY,
                temperature=0.5,
                model_name="gpt-3.5-turbo",
            )
            self.chatGPT = ChatGPT()
            self.chatGPT.chainInitializer(self.llm)

        except ValidationError:
            pass

    def classifyTaskCategory(self, user_input):
        try:
            return self.chatGPT.classifyTaskCategory(user_input)
        except AttributeError:
            return {
                "error_occurred": True,
                "response": None,
                "error": "OpenAI API Key is missing.",
            }

    def generalAssistant(self, user_input, chatHistory):
        return self.chatGPT.generalAssistant(user_input, chatHistory)
