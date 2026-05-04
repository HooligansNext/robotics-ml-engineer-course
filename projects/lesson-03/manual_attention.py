"""
Scaled dot-product attention from scratch.
Given queries Q, keys K, values V:
    attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
"""
import numpy as np

def softmax(x, axis=-1):
    """Numerically stable softmax."""
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)        # alignment between every query and every key
    weights = softmax(scores, axis=-1)      # turn alignments into probabilities
    return weights @ V, weights              # weighted sum of values + the weights themselves

# Tiny example: 3 tokens, each represented by a 4-dim vector
np.random.seed(0)
Q = np.random.randn(3, 4)
K = np.random.randn(3, 4)
V = np.random.randn(3, 4)

output, weights = scaled_dot_product_attention(Q, K, V)
print("Attention output shape:", output.shape)
print("\nAttention weights (each row sums to 1):")
print(np.round(weights, 3))
print("\nRow sums:", weights.sum(axis=1))
print("\nAttention output:")
print(np.round(output, 3))