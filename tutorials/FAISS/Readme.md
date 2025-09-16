We have a small database of 3 points (called `xb`) and one query `q`:

```makefile
xb[0] = [1, 1]
xb[1] = [2, 2]
xb[2] = [5, 4]

q     = [2, 1]
k     = 2   # we want the 2 nearest neighbours
```

Step 1 — find how far `q` is from each `xb[i]`

We measure “how far” by doing this simple recipe for each database point:

1.  Subtract the coordinates (component-wise): `(q - xb[i])`.
2.  Square each subtraction (so negative becomes positive).
3.  Add the squared numbers — that sum is the squared distance.

Do it for each:

For xb[0] = [1,1]

- Subtract: `q - xb[0] = [2,1] - [1,1] = [2-1, 1-1] = [1, 0]`
- Square each: `[1^2, 0^2] = [1, 0]`
- Add: `1 + 0 = 1`
  - squared distance = 1

For xb[1] = [2,2]

- Subtract: `q - xb[1] = [2,1] - [2,2] = [0, -1]`
- Square each: `[0^2, (-1)^2] = [0, 1]`
- Add: `0 + 1 = 1`
  - squared distance = 1

For xb[2] = [5,4]

- Subtract: `q - xb[2] = [2,1] - [5,4] = [-3, -3]`
- Square each: `[(-3)^2, (-3)^2] = [9, 9]`
- Add: `9 + 9 = 18`
  - squared distance = 18

So the three squared distances are:

```csharp
[1, 1, 18]
```

Pick the smallest k = 2 distances

## Why FAISS returns squared distances (not sqrt)

- Squared distance = sum of squared component differences.
- Sorting by squared distance gives the same order as sorting by real distance (because sqrt is monotonic).
- Why do this? sqrt is slower to compute; skipping it makes comparisons faster. So FAISS returns squared distances for speed.

## Where FAISS uses the identity `||q-x||^2 = ||q||^2 + ||x||^2 - 2q.x` ?

FAISS uses the identity everywhere it needs to compute many Euclidean distances fast.

1.  Exact (flat) search (IndexFlatL2):

    - When you call `index.search(xq, k)` FAISS often computes distances between `nq` queries and `nb` database vectors in one or a few big operations.
    - Instead of computing each `∑(q_j - x_j)^2` separately, FAISS computes:
      - `q_norm2` for each query (`∥q∥²`),
      - `x_norm2` for every database vector (`∥x∥²`) — precomputed and stored once,
      - `dot = Q @ X^T` (a matrix of dot products between all queries and all database vectors) using a highly-optimized matrix multiply (`BLAS/GEMM`).
      - Then it builds the distance matrix as `D = q_norm2[:,None] + x_norm2[None,:] - 2*dot`. This uses the identity and is much faster because big matrix multiplies are highly optimized.

2.  Batching & vectorization:

    - Computing `Q @ X^T` is done in large blocks so CPU/GPU vectorization and multi-threading are used. This is far faster than many tiny Python loops.

### Tiny numeric example that mirrors the trick

We’ll reuse your tiny example `q=[2,1]` and `xb = [[1,1],[2,2],[5,4]]`.

Compute the pieces:

- `∥q∥² = 2^2 + 1^2 = 5`
- `∥xb[0]∥² = 1^2 + 1^2 = 2`
- `∥xb[1]∥² = 2^2 + 2^2 = 8`
- `∥xb[2]∥² = 5^2 + 4^2 = 41`

- Dot products q·xb[i]:

  - `q·xb[0] = 2*1 + 1*1 = 3`
  - `q·xb[1] = 2*2 + 1*2 = 6`
  - `q·xb[2] = 2*5 + 1*4 = 14`

Now use the identity

- With `xb[0]`: `5 + 2 - 2*3 = 7 - 6 = 1`
- With `xb[1]`: `5 + 8 - 2*6 = 13 - 12 = 1`
- With `xb[2]`: `5 + 41 - 2*14 = 46 - 28 = 18`

Same results as doing `(2-1)^2+(1-1)^2` etc.
But now notice: if you have many queries, you can compute all dot products `Q @ X^T` (a matrix) with one call and then add precomputed norms — super efficient.

`dot = Q @ X.T` is a matrix multiplication (also called a matrix product), and every entry of that product is the dot product between one query vector and one database vector.
