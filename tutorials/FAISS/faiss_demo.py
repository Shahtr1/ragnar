# This script creates a random database of 10,000 vectors and 5 query vectors, 
# builds an exact L2 (Euclidean) FAISS index that stores the database, 
# then finds the 5 nearest database vectors for each query. 
# The output I is the indices of those neighbors and D are the squared L2 distances.

import faiss
import numpy as np

# 1. Create some random vectors (database)
d = 64   # dimension
nb = 10000  # database size
nq = 5      # number of queries
np.random.seed(42)

xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')

# 2. Build the index
index = faiss.IndexFlatL2(d)  # L2 distance
# IndexFlatL2(d) creates an exact, flat index that uses L2 (Euclidean) distance. 
# “Flat” means it stores all vectors and does not compress or approximate them.

index.add(xb)                 # add database vectors

# 3. Search
D, I = index.search(xq, 5)  # search top-5 nearest
print(I)  # indices of nearest neighbors
print(D)  # distances


