<div align="center">

<h1 style="border-bottom: none">
    <b><a href="https://github.com/pinacai">PINAC</a></b><br>
    Personal Intelligent Network Assistant Companion
</h1>

<img src="https://github.com/pinacai/NexusGPT/blob/main/assets/header_image.png" alt="header image">

<br>
<br>

[![](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com)
[![](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)

A simple-to-use framework for getting high-quality text for specific tasks using AI.  
Uplifting the AI-generated Text Quality

</div>

<br />

## What is PINAC-Nexus ?

In today's AI-driven world, many users find themselves caught between the desire for high-quality results and the complexity of crafting elaborate prompts. NexusGPT bridges this gap, catering to those who seek exceptional outcomes without the need for extensive prompt engineering skills.

Our innovative terminal tool simplifies the interaction process, allowing users to communicate their needs using concise, natural language. By leveraging advanced AI algorithms, NexusGPT interprets these simple inputs and generates optimal results, eliminating the barrier of complex prompt creation.

By democratizing access to powerful AI capabilities, NexusGPT empowers users across various domains to harness the full potential of language models. This tool not only saves time and reduces frustration but also opens up new possibilities for creativity and problem-solving, making advanced AI assistance accessible to a broader audience.

Inspired by <a href="https://github.com/danielmiessler/fabric">Fabric</a> but making it even better.

## 🏛️ Architecture

The main idea of NexusGPT is to apply high-quality prompts to the query given by the user. But the main challenge is to identify the task for applying the prompt, for it you can use specific commands like <a href="https://github.com/danielmiessler/fabric">Fabric</a>. But it's hard to remember all those commands. So I used GPTs feed with the custom dataset to able to identify the task category. Now it applies the high-quality prompt to get the best-in-class text from AI Models.

Currently, this tool has the option of GPT 3.5 and Gemini models, we are actively working to add all the available LLMs.

<img src="https://github.com/pinacai/NexusGPT/blob/main/assets/Architecture_Diagram.svg" alt="Architecture Diagram">


## 🚀 Quickstart

> It's always recommended to use a virtual environment

1. Download the project and navigate to its directory
   ```bash
   git clone https://github.com/pinacai/NexusGPT.git && cd NexusGPT
   ```

2. Install pip dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Set up API keys by storing them in a .env file in the project root folder. You can store the API Key of the AI Model you want to use.
   ```
   OPENAI_API_KEY={Your OpenAI API Key here}
   GOOGLE_API_KEY={Your Gemini API Key here}
   ```

## 🎉 Contributing

1. Star the repository
2. Fork the repository on GitHub.
3. Clone the project to your machine.
4. Commit changes to your branch.
5. Push your work back up to your fork.
6. Submit a Pull request so that we can review your changes

## 📝 License

PINAC Workspace is licensed under the **MIT license**. See the <a href="https://github.com/pinacai/NexusGPT/blob/main/LICENSE">LICENSE</a> file for more details.

## 🌐 Support

If you have any support questions or to report issues, please file an issue through the GitHub issue tracker associated with the repository.

## 🧑‍💻 About Us

A GitHub organization committed to creating AI-powered applications that address practical problems, making AI accessible to everyone.


![Line](https://user-images.githubusercontent.com/85225156/171937799-8fc9e255-9889-4642-9c92-6df85fb86e82.gif)

<div align="center">
  <h1>Tip from us 😇</h1>
  <p>It always takes time to understand and learn. So, don't worry at all. We know <b>you have got this</b>! 💪</p>
  <h3>Show some &nbsp;❤️&nbsp; by &nbsp;🌟&nbsp; this repository!</h3>
</div>
