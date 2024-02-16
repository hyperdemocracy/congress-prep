from nomic import atlas

#project_id = XYZ
#ds = atlas.AtlasDataset(dataset_id=project_id)

project_name = "gabrielhyperdemocracy/us-congressional-legislation"
ds = atlas.AtlasDataset(identifier=project_name)

map = ds.maps[0]

#df_projected_vecs = map.embeddings.projected
#df_latent_vecs = map.embeddings.latent
#df_topics = map.topics.df

chunk_id = "114-s-1192-is-1"
with ds.wait_for_dataset_lock():
  neighbors, dists = map.embeddings.vector_search(ids=["114-s-1192-is-1"], k=5)

# Return similar data points
similar_datapoints = ds.get_data(ids=neighbors[0])

print(similar_datapoints)
