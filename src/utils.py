import pandas as pd
import numpy as np
from tqdm import tqdm
import tiktoken
from openai import OpenAI
import re

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


def get_embedding_query(readme_topic_df: pd.DataFrame) -> pd.DataFrame:
    readme_topic_df = readme_topic_df.copy()
    readmes = readme_topic_df['readme'].values
    topics = readme_topic_df['topic'].values

    # readme
    readme_embeddings = []
    for r in tqdm(readmes):
        response = OpenAI().embeddings.create(
            input=TokenUtils(r).trim(),
            model='text-embedding-3-small'
        )
        readme_embeddings.append(response.data[0].embedding[:512])
    readme_topic_df['readme_embeddings'] = readme_embeddings

    # topics (need agg)
    topic_embeddings = []
    for t in tqdm(topics):
        if not any(t):
            topic_embeddings.append(None)
            continue
        response = OpenAI().embeddings.create(
            input=t,
            model='text-embedding-3-small'
        )
        agg_emb = np.mean([i.embedding for i in response.data], axis=0)
        topic_embeddings.append(agg_emb[:512])
    readme_topic_df['topic_embeddings'] = topic_embeddings
    return readme_topic_df