
import requests
from typing import List

def get_embedding(text_list: List[str]):
    response = requests.post("http://65.0.109.171:5000/embed", json={
        "texts": text_list,
        "chunk": True
    })

    if response.status_code == 200:
        return response.json().get("embeddings")
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

