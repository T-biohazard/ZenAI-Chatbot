import gradio as gr
import requests
import os

def chat_with_backend(message, history):
    try:
        formatted_history = []
        for conversation in history:
            if isinstance(conversation, list) and len(conversation) == 2:
                user_msg, bot_msg = conversation
                formatted_history.append([user_msg, bot_msg])
        
        backend_url = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
        response = requests.post(
            f"{backend_url}/chat", 
            json={
                "message": message,
                "history": formatted_history
            }
        )
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"Error: {response.status_code} - {response.text}"
            
    except requests.exceptions.ConnectionError:
        return "Connection error - backend service unavailable"
    except Exception as e:
        return f"Error: {str(e)}"

demo_interface = gr.ChatInterface(
    fn=chat_with_backend,
    title="ZenAI",
    description="AI Assistant"
)

if __name__ == "__main__":
    demo_interface.launch(
        debug=True, 
        share=False, 
        server_port=7860,
        server_name="0.0.0.0"  
    )