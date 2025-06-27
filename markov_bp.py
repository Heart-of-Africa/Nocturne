
import torch

def compute_transition_matrix(h):
    raw = torch.outer(h, h)
    M = torch.softmax(raw, dim=-1)
    return M

def markov_backprop(grads, states):
    T = len(grads)
    for t in reversed(range(T - 1)):
        M = compute_transition_matrix(states[t])
        grads[t] = M @ grads[t + 1]
    return grads
