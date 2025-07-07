import pandas as pd
import numpy as np
from tqdm import tqdm
import tiktoken
from openai import OpenAI
import re
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
import json

class TokenUtils:
    def __init__(self, text):
        if not text:
            self.text = " "
        else:
            self.text = text
        self.encoding = tiktoken.encoding_for_model('text-embedding-ada-002')
        self.tokens = self.encoding.encode(self.text)
        self.token_count = len(self.tokens)

    def trim(self, size=8000):
        if self.token_count > size:
            trimmed_tokens = self.tokens[:size]
            return self.encoding.decode(trimmed_tokens)
        return self.text


def filter_badges(text):
    # remove status badges in github readme
    # this usually contains lots of tests therefore cause false positives for test/quality categorization
    if not text:
        return text
    phrases = ('![', '[![')
    return '\n'.join([i for i in text.split('\n') if not i.startswith(phrases)])


def filter_css(text):        
    # removing CSS
    if not text:
        return text
    pattern = re.compile(r'<[^>\n]+(?:>|$)')
    cleaned_text = pattern.sub('', text)
    return cleaned_text



def calc_kw_distance(query_array: List[List[float]], kw_emb_df: pd.DataFrame):   

    if 'embeddings' not in kw_emb_df.columns:
        raise ValueError("kw_emb_df must have an 'embeddings' column.")
    if not query_array:
        raise ValueError("query_array must not be empty.")
    
    df = kw_emb_df.copy()
    distances = cosine_similarity(
        np.vstack(df['embeddings'].tolist()), 
        np.vstack(query_array)
    )
    if distances.shape[1] > 1:
        # print('Multiple queries found, averaging similartiy...')
        distances = np.mean(distances, axis=1)
    
    df['distance'] = distances
    return df.drop(['embeddings'], axis=1)
    

def output_json(df):
    output = {}
    output['winner'] = df[df['grouping'] == df['grouping'].min()].category_name.tolist()
    output['prediction_cluster_raw'] = df.groupby(df['grouping'].astype(int))['category_name'].agg(list).to_dict()
    output['prediction_distance_raw'] = {k:v for k, v in df[['category_name', 'distance']].values}
    return json.dumps(output, indent=2)


def get_emb_batch(batch_text: List[str], emb_size: int=512) -> List[List[float]]:
    """Generate embeddings for a batch of text strings."""

    try:
        response = OpenAI().embeddings.create(
            input=[TokenUtils(input_text).trim() for input_text in batch_text],
            model='text-embedding-3-small'
        )
        return [data.embedding[:emb_size] for data in response.data]

    except Exception as e:
        raise RuntimeError(f"Failed to get embeddings: {e}")    


def get_kw_embedding_df(kw: dict) -> pd.DataFrame:
    """convert defined categories and their keywords to df with embeddings"""

    kw_df = pd.DataFrame(
        [(k, v) for k, values in component_keywords.items() for v in values], 
        columns=["category_name", "keywords"]
    )

    # calculate emb
    kw_df['embeddings'] = get_emb_batch(kw_df["keywords"].values)

    return kw_df


# def get_embedding_query(readme_topic_df: pd.DataFrame) -> pd.DataFrame:
#     readme_topic_df = readme_topic_df.copy()
#     readmes = readme_topic_df['readme'].values
#     topics = readme_topic_df['topic'].values

#     # readme
#     readme_embeddings = []
#     for r in tqdm(readmes):
#         response = OpenAI().embeddings.create(
#             input=TokenUtils(r).trim(),
#             model='text-embedding-3-small'
#         )
#         readme_embeddings.append(response.data[0].embedding[:512])
#     readme_topic_df['readme_embeddings'] = readme_embeddings

#     # topics (need agg)
#     topic_embeddings = []
#     for t in tqdm(topics):
#         if not any(t):
#             topic_embeddings.append(None)
#             continue
#         response = OpenAI().embeddings.create(
#             input=t,
#             model='text-embedding-3-small'
#         )
#         agg_emb = np.mean([i.embedding for i in response.data], axis=0)
#         topic_embeddings.append(agg_emb[:512])
#     readme_topic_df['topic_embeddings'] = topic_embeddings
#     return readme_topic_df