import numpy as np

# database (nb x d) and query (nq x d)
xb = np.array([[1,1],[2,2],[5,4]], dtype=np.float32)   # nb=3, d=2
q  = np.array([[2,1]], dtype=np.float32)               # nq=1, d=2

# precompute squared norms
x_norm2 = np.sum(xb * xb, axis=1)   # shape (nb,)
q_norm2 = np.sum(q * q, axis=1)    # shape (nq,)

# compute dot products between all queries and db vectors
dot = q @ xb.T                      # shape (nq, nb)

# squared distance matrix using identity
D = q_norm2[:, None] + x_norm2[None, :] - 2.0 * dot   # shape (nq, nb)

print("squared distances:", D)   # -> [[1., 1., 18.]]