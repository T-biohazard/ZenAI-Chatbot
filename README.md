# Chatbot
A Chatbot with LangChain, LangGraph, Groq LLM, and multiple tools like Arxiv, Wikipedia, Tavily, and crawl4ai. 

# Installation Instructions
1. Clone the Repository
git clone https://github.com/T-biohazard/zenai.git
2. Environment Setup
- Create a .env file in the project root and add 2 keys : 
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

3. Docker Deployment
# Build and run with Docker Compose
docker-compose up --build

# Or can also run in detached mode
docker-compose up -d --build

4. Local Development
Open terminal and add these commands : 
pip install -r requirements.txt - install dependencies 

python backend.py - for backend 

python frontend.py - for frontend
