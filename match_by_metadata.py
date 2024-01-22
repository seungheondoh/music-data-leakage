import os
import argparse
import torch
import numpy as np
import pandas as pd
from sentence_transformers import util

def load_pretrain_db_embs(db_name="fma"):
    db_embs = torch.load(f"../embeddings/{db_name}_embs.pt")
    z_db = torch.from_numpy(np.stack([i["embeddings"] for i in db_embs.values()]))
    idx_db = [i["track_id"] for i in db_embs.values()]
    q_db = [i["query"] for i in db_embs.values()]
    return z_db, idx_db, q_db
    
def load_downstream_query_embs(query_name="emo"):
    q_embs = torch.load(f"../embeddings/{query_name}_embs.pt")
    z_query = torch.from_numpy(np.stack([i["embeddings"] for i in q_embs.values()]))
    idx_query = [i["track_id"] for i in q_embs.values()]
    q_query = [i["query"] for i in q_embs.values()]
    return z_query, idx_query, q_query
    
def load_testset():
    return None

def main(args):
    z_db, idx_db, q_db = load_pretrain_db_embs(db_name=args.db_name)
    z_query, idx_query, q_query = load_downstream_query_embs(query_name=args.query_name)
    cos_sim = util.cos_sim(z_query, z_db)
    results = []
    for idx, sim in enumerate(cos_sim):
        max_idx = sim.argmax()
        results.append({
            "query_track_id":idx_query[idx],
            "target_track_id": idx_db[max_idx],
            "score": float(sim[max_idx]),
            "query_text": q_query[idx],
            "target_text": q_db[max_idx],
        })
    df = pd.DataFrame(results)
    df_threshold = df[df['score'] > 0.8]
    test_track_ids = load_testset()
    if test_track_ids:
        overlap_set = set(df_threshold["query_track_id"])
        overlap_test = test_track_ids.intersection(overlap_set)
        df_threshold = df_threshold.set_index("query_track_id").loc[list(overlap_test)].reset_index()
    df_threshold.to_csv(f"./overlap/{args.db_name}_{args.query_name}.csv")
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--db_name', type=str, default="pretrain")
    parser.add_argument('--query_name', type=str, default="pretrain")
    args = parser.parse_args()
    main(args)