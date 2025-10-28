import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError('Google API key not found. Please set the GOOGLE_API_KEY environment variable.')

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7
)

session_memory_map = {}

def get_response(session_id: str, user_query: str) -> str:
    if session_id not in session_memory_map:
        memory = ConversationBufferMemory(return_messages=True)

        # Friendly + motivational style with emojis
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are a supportive, friendly AI buddy 😊. "
             "Always keep responses short (2–3 sentences), motivating, and positive. "
             "Feel free to use emojis like 🌟💪✨ when appropriate."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

        session_memory_map[session_id] = ConversationChain(
            llm=llm,
            memory=memory,
            prompt=prompt,
            verbose=True
        )

    conversation = session_memory_map[session_id]
    return conversation.predict(input=user_query)
