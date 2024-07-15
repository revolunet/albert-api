import sys
from fastapi.testclient import TestClient
sys.path.append("..")
from main import app  # Import the FastAPI app instance

user = "leo"
chat_id = "0e64d044-1c1f-4c49-832a-2b426efe4fa8"
model = "BAAI/bge-m3"

data = {
    "model": "AgentPublic/llama3-instruct-8b",
    "messages": [{"role": "user", "content": "Peut-on avoir des jours de congés pour un mariage ?"}],
    "stream": False,
    "n": 1,
    "user": user,
    "tools": [
        {
            "function": {
                "name": "MultiAgents",
                "parameters": {
                    "embeddings_model": model,
                    "chat_id": chat_id,
                },
            },
            "type": "function",
        }
    ],
}
with TestClient(app) as client:
    response = client.chat.completions.create(**data)
    assert response.status_code == 200
