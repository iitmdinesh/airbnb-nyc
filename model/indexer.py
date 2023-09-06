from typing import List

import pandas as pd
import numpy as np

from langchain.embeddings import HuggingFaceEmbeddings


def tabular_to_text_template(**kwargs):
    return f"""The property is of type {kwargs['property_type']}. The room is of type {kwargs['room_type']} and it can 
            accomodate {kwargs['accommodates']} people. The room has {kwargs['bathrooms_text']} bathrooms and 
            {kwargs['bedrooms']} bedrooms with a total of {kwargs['beds']} beds. The price per night of stay is 
            {kwargs['price']} with a stay requirement of {kwargs['minimum_nights']}-{kwargs['maximum_nights']} nights. 
            The place has the following description: {kwargs['description']}"""


def embed_text(
        texts: List[str],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device='mps',
):
    model_kwargs = {
        'device': device,
    }
    encode_kwargs = {
        'normalize_embeddings': True,
        'show_progress_bar': True,
        'batch_size': 512,
    }
    embed_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return np.array(embed_model.embed_documents(texts))


def create_index(df: pd.DataFrame):
    texts = []
    ids = []
    for data in df.to_dict(orient='records'):
        ids.append(data['id'])
        texts.append(tabular_to_text_template(**data))
    ids = np.array(ids, dtype=np.int64)
    embeds = embed_text(texts)
    return ids, embeds
