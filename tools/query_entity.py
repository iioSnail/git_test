import os

import torch
import torch.nn.functional as F

from tools.embedding_entity import embedding_entity
from utils.utils import load_obj


class QueryEntity(object):

    def __init__(self):
        self.entities, self.embeddings = self.load_entity_embeddings()

    def load_entity_embeddings(self):
        entity_embeddings = load_obj("entity_embeddings.pkl")

        entities = []
        embeddings = []
        for item in entity_embeddings:
            entities.append(item['entity'])
            embeddings.append(item['embedding'])

        embeddings = torch.vstack(embeddings)

        return entities, embeddings

    def query_entity(self, entity):
        embedding = embedding_entity(entity)
        sims = F.cosine_similarity(embedding, self.embeddings)
        index = sims.argmax(-1)

        if sims[index] < 0.95:
            return entity

        return self.entities[index]


if __name__ == '__main__':
    query_entity = QueryEntity()
    print(query_entity.query_entity("康达姆机器人"))
