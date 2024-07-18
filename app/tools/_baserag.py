from typing import List, Optional

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from fastapi import HTTPException
from qdrant_client.http import models as rest

from app.utils.security import secure_data
from app.utils.data import search_multiple_collections, get_all_collections


class BaseRAG:
    """
    Base RAG, basic retrival augmented generation.

    Args:
        embeddings_model (str): OpenAI embeddings model
        collection (Optional[List[str]], optional): List of collections to search in. Defaults to None (all collections).
        file_ids (Optional[List[str]], optional): List of file IDs in the selected collections (after upload files). Defaults to None (all files are selected).
        k (int, optional): Top K per collection (max: 6). Defaults to 4.
        prompt_template (Optional[str], optional): Prompt template. Defaults to DEFAULT_PROMPT_TEMPLATE.
    """

    DEFAULT_PROMPT_TEMPLATE = "Réponds à la question suivante en te basant sur les documents ci-dessous : %(prompt)s\n\nDocuments :\n\n%(docs)s"
    MAX_K = 6

    def __init__(self, clients: dict):
        self.clients = clients

    @secure_data
    async def get_prompt(
        self,
        embeddings_model: str,
        collections: Optional[List[str]] = None,
        file_ids: Optional[List[str]] = None,
        k: Optional[int] = 4,
        prompt_template: Optional[str] = DEFAULT_PROMPT_TEMPLATE,
        **request,
    ) -> str:
        if k > self.MAX_K:
            raise HTTPException(
                status_code=400, detail=f"K must be less than or equal to {self.MAX_K}"
            )
        
        embeddings = HuggingFaceEndpointEmbeddings(
            model=str(self.clients["openai"][embeddings_model].base_url),
            huggingfacehub_api_token=self.clients["openai"][embeddings_model].api_key,
        )

        filter = rest.Filter(must=[rest.FieldCondition(key="metadata.file_id", match=rest.MatchAny(any=file_ids))]) if file_ids else None  # fmt: off
        prompt = request["messages"][-1]["content"]

        all_collections = get_all_collections(vectorstore=self.clients["vectors"], api_key=request["api_key"])
        collections = collections or all_collections
        
        for collection in collections:
            if collection not in all_collections:
                raise HTTPException(status_code=404, detail="Collection not found.")
        
        docs = search_multiple_collections(
            vectorstore=self.clients["vectors"],
            emmbeddings=embeddings,
            prompt=prompt,
            collections=collections,
            k=k,
            filter=filter,
        )
        docs = "\n\n".join([doc.page_content for doc in docs])
        prompt = prompt_template % {"docs": docs, "prompt": prompt}

        return prompt
