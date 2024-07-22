import os
from models.__init__ import prepareDatasetWithoutSystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic.v1.error_wrappers import ValidationError

# Loading Local API Keys
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# output parser
output_parser = StrOutputParser()

# Datasets
TCC_Dataset = prepareDatasetWithoutSystemMessage("dataset/TCC_Dataset.csv")
CAIT_Dataset = prepareDatasetWithoutSystemMessage("dataset/CAIT_Dataset.csv")


class Gemini:
    def __init__(self):
        self.taskClassificationPrompt = ChatPromptTemplate.from_messages(TCC_Dataset)
        self.generalAssistantPrompt = ChatPromptTemplate.from_messages(CAIT_Dataset)

    def chainInitializer(self, llm):
        self.taskClassificationChain = (
            self.taskClassificationPrompt | llm | output_parser
        )
        self.generalAssistantChain = self.generalAssistantPrompt | llm | output_parser

    # For classifying the user's query into specific task category
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


# =================== GEMINI 1.5 PRO =================== #


class Gemini_1_5_Pro:
    def __init__(self):
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro-latest", api_key=GOOGLE_API_KEY
            )
            self.Gemini = Gemini()
            self.Gemini.chainInitializer(self.llm)

        except ValidationError:
            pass

    def classifyTaskCategory(self, user_input):
        try:
            return self.Gemini.classifyTaskCategory(user_input)
        except AttributeError:
            return {
                "error_occurred": True,
                "response": None,
                "error": "GEMINI API Key is missing.",
            }

    def generalAssistant(self, user_input, chatHistory):
        return self.Gemini.generalAssistant(user_input, chatHistory)


# =================== GEMINI 1.0 PRO =================== #


class Gemini_1_Pro:
    def __init__(self):
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.0-pro", api_key=GOOGLE_API_KEY
            )
            self.Gemini = Gemini()
            self.Gemini.chainInitializer(self.llm)

        except ValidationError:
            pass

    def classifyTaskCategory(self, user_input):
        try:
            return self.Gemini.classifyTaskCategory(user_input)
        except AttributeError:
            return {
                "error_occurred": True,
                "response": None,
                "error": "GEMINI API Key is missing.",
            }

    def generalAssistant(self, user_input, chatHistory):
        return self.Gemini.generalAssistant(user_input, chatHistory)


# =================== GEMINI 1.5 FLASH =================== #


class Gemini_1_5_Flash:
    def __init__(self):
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash-latest", api_key=GOOGLE_API_KEY
            )
            self.Gemini = Gemini()
            self.Gemini.chainInitializer(self.llm)

        except ValidationError:
            pass

    def classifyTaskCategory(self, user_input):
        try:
            return self.Gemini.classifyTaskCategory(user_input)
        except AttributeError:
            return {
                "error_occurred": True,
                "response": None,
                "error": "GEMINI API Key is missing.",
            }

    def generalAssistant(self, user_input, chatHistory):
        return self.Gemini.generalAssistant(user_input, chatHistory)
