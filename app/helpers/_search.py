from typing import List

from app.helpers._internetclient import InternetClient
from app.helpers.searchclients import SearchClient
from app.schemas.search import Search
from app.schemas.security import User
from app.utils.variables import INTERNET_COLLECTION_DISPLAY_ID


class Search:
    def __init__(self, engine_client: SearchClient, internet_client: InternetClient):
        self.engine_client = engine_client
        self.internet_client = internet_client

    def query(self, 
        collections: List[str], 
        prompt: str, 
        method: str, 
        k: int, 
        rff_k: int, 
        score_threshold: float, 
        user: User
    ) -> List[Search]:
        need_internet_search = not collections or INTERNET_COLLECTION_DISPLAY_ID in collections
        internet_chunks = []
        if need_internet_search:
            internet_chunks = self.internet_client.get_chunks(prompt=prompt)

            if internet_chunks:
                internet_collection = self.internet_client.create_temporary_internet_collection(internet_chunks, collections, user)

            if INTERNET_COLLECTION_DISPLAY_ID in collections:
                collections.remove(INTERNET_COLLECTION_DISPLAY_ID)
                if not collections and not internet_chunks:
                    return []
                if internet_chunks:
                    collections.append(internet_collection.id)

        searches = self.engine_client.query(
            prompt=prompt,
            collection_ids=collections,
            method=method,
            k=k,
            rff_k=rff_k,
            score_threshold=score_threshold,
            user=user,
        )

        if internet_chunks:
            self.engine_client.delete_collection(internet_collection.id, user=user)

        return searches



        '''
        data = []
        if not collections or INTERNET_COLLECTION_ID in collections:
            if INTERNET_COLLECTION_ID in collections:
                collections.remove(INTERNET_COLLECTION_ID)
            internet = SearchOnInternet(models=self.vectors.models)
            if len(collections) > 0:
                collection_model = self.vectors.get_collections(collection_ids=collections, user=user)[0].model
            else:
                collection_model = None
            data.extend(internet.search(prompt=prompt, n=4, model_id=collection_model, score_threshold=score_threshold))

        if len(collections) > 0:
            data.extend(self.vectors.search(prompt=prompt, collection_ids=collections, k=k, score_threshold=score_threshold, user=user))

        data = sorted(data, key=lambda x: x.score, reverse=False)[:k]

        return data
        '''
