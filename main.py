import pyfiglet
import inquirer
import colorama
from colorama import Fore, Style
from models.ChatGPT import ChatGPT_3_5
from models.Gemini import Gemini_1_5_Pro, Gemini_1_Pro, Gemini_1_5_Flash
from prompts.prompt import givePrompt
from langchain.schema import HumanMessage, AIMessage


# Initializing AI Models
chatgpt_3_5 = ChatGPT_3_5()
gemini_1_5_pro = Gemini_1_5_Pro()
gemini_1_pro = Gemini_1_Pro()
gemini_1_5_flash = Gemini_1_5_Flash()

# Initialize colorama
colorama.init()

# Initialize the chat history
chatHistory = []


def createResponse(AiModel, query, prompt_name, response_type):
    # gets the required prompt
    prompt = givePrompt(prompt_name, query)
    ai_response = AiModel.generalAssistant(prompt, chatHistory)
    if not ai_response["error_occurred"]:
        chatHistory.append(AIMessage(content=ai_response["response"]))
        return {
            "error_occurred": False,
            "response": {
                "type": response_type,
                "content": ai_response["response"],
            },
            "error": None,
        }
    else:
        return ai_response


def giveAiResponse(preferred_model, query):
    # Initializing AI Models
    if preferred_model == "ChatGPT-3.5 turbo":
        AiModel = chatgpt_3_5
    elif preferred_model == "Gemini 1.5 Pro":
        AiModel = gemini_1_5_pro
    elif preferred_model == "Gemini 1.0 Pro":
        AiModel = gemini_1_pro
    elif preferred_model == "Gemini Flash 1.5":
        AiModel = gemini_1_5_flash

    chatHistory.append(HumanMessage(content=query))
    ai_response = AiModel.classifyTaskCategory(query)

    if not ai_response["error_occurred"]:
        response = createResponse(AiModel, query, "others", "others")
    else:
        response = ai_response

    return response


if __name__ == "__main__":
    # Printig Header
    ascii_banner = pyfiglet.figlet_format("NexusGPT", font="standard")
    colored_banner = Fore.CYAN + Style.BRIGHT + ascii_banner + Style.RESET_ALL
    print(colored_banner)

    questions = [
        inquirer.List(
            "model",
            message="What AI Model You Want to Use?",
            choices=[
                "ChatGPT-3.5 turbo",
                "Gemini 1.0 Pro",
                "Gemini 1.5 Pro",
                "Gemini Flash 1.5",
            ],
        ),
    ]
    answer = inquirer.prompt(questions)
    print(answer["model"])
    print("type '/bye' to exit")
    print()
    while True:
        query = input("\npinac-nexus> ")
        if query == "/bye":
            break
        else:
            response = giveAiResponse(answer["model"], query)
            if not response["error_occurred"]:
                print(response["response"]["content"])
            else:
                print(response["error"])
