<div align="center">

<h1 style="border-bottom: none">
    <b><a href="https://github.com/pinacai">PINAC</a></b><br>
    Uplifting the AI-generated Text Quality
</h1>

<img src="https://github.com/pinacai/NexusGPT/blob/main/assets/header_image.png" alt="header image">

<br>
<br>

[![](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com)
[![](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)

A simple-to-use framework for getting high-quality text for specific tasks using AI.  
Taking Quality-Text Generation to Next Level

</div>

<br />

## Why PINAC-Nexus ?

Maybe you are like me, neither a prompt engineer nor like to write long prompts for a simple task but still, we deserve the best result. So, I made this terminal tool so that you can tell your need in short and simple words and this tool will generate the best result for you!

Inspired by <a href="https://github.com/danielmiessler/fabric">Fabric</a> but making it even better, removing the need to remember particular commands, making it simple to use.


## 🏛️ Architecture

The main idea of NexusGPT is to apply high quality prompt on the query given by user. But the main challenges is to identify the task for applying prompt, for it you can use specific command like <a href="https://github.com/danielmiessler/fabric">Fabric</a>. But it's headeach to remember all those command. So I used GPTs feed with the custom dataset to able to identify the task category. Now it applyes the high quality prompt to get the best in class text from AI Models.

Currenyly this tool have option of GPT 3.5 and Gimini models, we are actively working to add all the available LLMs.

<img src="https://github.com/pinacai/NexusGPT/blob/main/assets/Architecture_Diagram.svg" alt="Architecture Diagram">


## 🚀 Quickstart

> It's always recommended to use virtual environment

1. Download the project and navigate to its directory
   ```bash
   git clone https://github.com/pinacai/NexusGPT.git && cd NexusGPT
   ```

2. Install pip dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Set up API keys by storing them in a .env file in the project root folder. Store the API Key of the AI Model you want to use.
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

**Maintainer**: @RajeshTechForge

![Line](https://user-images.githubusercontent.com/85225156/171937799-8fc9e255-9889-4642-9c92-6df85fb86e82.gif)

<div align="center">
  <h1>Tip from us 😇</h1>
  <p>It always takes time to understand and learn. So, don't worry at all. We know <b>you have got this</b>! 💪</p>
  <h3>Show some &nbsp;❤️&nbsp; by &nbsp;🌟&nbsp; this repository!</h3>
</div>
