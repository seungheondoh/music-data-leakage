import os
import ast
import json
import jsonlines
import torch
import pandas as pd
import sqlite3
import requests
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')
model = model.to("cuda:0")

def get_msd_metadata(db_path = "../dataset/msd/track_metadata.db"):
    con = sqlite3.connect(db_path)
    msd_db = pd.read_sql_query("SELECT * FROM songs", con)
    # msd_db = msd_db.set_index("track_id")
    return msd_db

def get_mtt_embs():
    mtt_df = pd.read_csv("./datasets/magnatagatune/clip_info_final.csv", sep="\t")
    mtt_df["query"] = mtt_df['title'] + [" by " for _ in range(len(mtt_df))] + mtt_df['artist'] + [" from " for _ in range(len(mtt_df))] + mtt_df['album']
    mtt_df = mtt_df[['clip_id','query']]
    mtt_df = mtt_df.groupby('query')['clip_id'].agg(list).reset_index()
    results_dict = {}
    batch_size = 128
    epoch = (len(mtt_df) // batch_size) + 1
    for i in tqdm(range(epoch)):
        batch = mtt_df.iloc[i*batch_size:(i+1)*batch_size]
        queries = list(batch['query'])
        clip_ids = list(batch['clip_id'])
        embeddings = model.encode(queries, convert_to_tensor=True)
        for query, clip_id, embs in zip(queries, clip_ids, embeddings):
            results_dict[query] = {
                'query': query,
                "clip_id": clip_id,
                "embeddings": embs.detach().cpu().numpy()
            }
    print(len(mtt_df), len(results_dict))
    torch.save(results_dict, "./embeddings/mtt_embs.pt")
    
def get_msd_embs():
    msd_db = get_msd_metadata()
    msd_db["query"] = msd_db['title'] + [" by " for _ in range(len(msd_db))] + msd_db['artist_name'] + [" from " for _ in range(len(msd_db))] + msd_db['release']
    msd_db = msd_db[["track_id",'query']]
    results_dict = {}
    batch_size = 2**13
    epoch = (len(msd_db) // batch_size) + 1
    for i in tqdm(range(epoch)):
        batch = msd_db.iloc[i*batch_size:(i+1)*batch_size]
        queries = list(batch['query'])
        track_ids = list(batch['track_id'])
        embeddings = model.encode(queries, convert_to_tensor=True)
        for query, track_id, embs in zip(queries, track_ids, embeddings):
            results_dict[query] = {
                'query': query,
                "track_id": track_id,
                "embeddings": embs.detach().cpu().numpy()
            }
    print(len(msd_db), len(results_dict))
    torch.save(results_dict, "./embeddings/msd_embs.pt")
    
def get_m4a_embs():
    dataset = load_dataset("hf_m4a_path")
    m4a_df = pd.DataFrame(dataset['train'])
    print(m4a_df.iloc[0])
    m4a_df["query"] = m4a_df['title'] + [" by " for _ in range(len(m4a_df))] + m4a_df['artist_name'] + [" from " for _ in range(len(m4a_df))] + m4a_df['release']
    m4a_df = m4a_df[['track_id','query']]
    results_dict = {}
    batch_size = 2**12
    epoch = (len(m4a_df) // batch_size) + 1
    for i in tqdm(range(epoch)):
        batch = m4a_df.iloc[i*batch_size:(i+1)*batch_size]
        queries = list(batch['query'])
        track_ids = list(batch['track_id'])
        embeddings = model.encode(queries, convert_to_tensor=True)
        for query, track_id, embs in zip(queries, track_ids, embeddings):
            results_dict[query] = {
                'query': query,
                "track_id": track_id,
                "embeddings": embs.detach().cpu().numpy()
            }
    print(len(m4a_df), len(results_dict))
    torch.save(results_dict, "./embeddings/m4a_embs.pt")
    
def get_gtzan_embs():
    dataset = load_dataset("seungheondoh/gtzan-bind")
    gtzan_df = pd.DataFrame(dataset['gtzan_bind_v1'])  
    gtzan_df["query"] = gtzan_df['title'] + [" by " for _ in range(len(gtzan_df))] + gtzan_df['artist_name'] + [" from " for _ in range(len(gtzan_df))] + gtzan_df['album']
    gtzan_df = gtzan_df[['track_id','query']]
    results_dict = {}  
    batch_size = 32
    epoch = (len(gtzan_df) // batch_size) + 1
    for i in tqdm(range(epoch)):
        batch = gtzan_df.iloc[i*batch_size:(i+1)*batch_size]
        queries = list(batch['query'])
        track_ids = list(batch['track_id'])
        embeddings = model.encode(queries, convert_to_tensor=True)
        for query, track_id, embs in zip(queries, track_ids, embeddings):
            results_dict[query] = {
                'query': query,
                "track_id": track_id,
                "embeddings": embs.detach().cpu().numpy()
            }
    print(len(gtzan_df), len(results_dict))
    torch.save(results_dict, "./embeddings/gtzan_embs.pt")

   
def get_fma_embs():
    dataset = load_dataset("hf_fma_path")
    fma_df = pd.DataFrame(dataset['train'])  
    print(fma_df.head())
    fma_df["query"] = fma_df['title'] + [" by " for _ in range(len(fma_df))] + fma_df['artist_name']
    fma_df = fma_df[['track_id','query']]
    results_dict = {}  
    batch_size = 2**13
    epoch = (len(fma_df) // batch_size) + 1
    for i in tqdm(range(epoch)):
        batch = fma_df.iloc[i*batch_size:(i+1)*batch_size]
        queries = list(batch['query'])
        track_ids = list(batch['track_id'])
        embeddings = model.encode(queries, convert_to_tensor=True)
        for query, track_id, embs in zip(queries, track_ids, embeddings):
            results_dict[query] = {
                'query': query,
                "track_id": track_id,
                "embeddings": embs.detach().cpu().numpy()
            }
    print(len(fma_df), len(results_dict))
    torch.save(results_dict, "./embeddings/fma_embs.pt")
    
def get_emo_embs():
    df_emo = pd.read_csv("./datasets/emo/songs_info.csv")
    df_emo['query'] = df_emo['Song title'] + [" by " for _ in range(len(df_emo))] + df_emo['Artist']
    df_emo['query'] = [i.replace("\t", "") for i in df_emo['query']]
    df_emo = df_emo[['song_id','query']]
    results_dict = {}  
    batch_size = 2**9
    epoch = (len(df_emo) // batch_size) + 1
    for i in tqdm(range(epoch)):
        batch = df_emo.iloc[i*batch_size:(i+1)*batch_size]
        queries = list(batch['query'])
        song_ids = list(batch['song_id'])
        embeddings = model.encode(queries, convert_to_tensor=True)
        for query, song_id, embs in zip(queries, song_ids, embeddings):
            results_dict[query] = {
                'query': query,
                "track_id": song_id,
                "embeddings": embs.detach().cpu().numpy()
            }
    print(len(df_emo), len(results_dict))
    torch.save(results_dict, "./embeddings/emo_embs.pt")

def main():
    get_mtt_embs()
    get_msd_embs()
    # get_m4a_embs()
    # get_fma_embs()
    get_gtzan_embs()
    get_emo_embs()
    


if __name__ == "__main__":
    main()