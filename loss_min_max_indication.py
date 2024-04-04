# %%
import math

N = 256
temp = 0.5


def nt_xent(sim_positives, sim_negatives, temp, N):
    numerator = math.exp(sim_positives / temp)
    denominator = math.exp(sim_negatives / temp) * 2 * (N - 1)

    l_nt_xent = -math.log(numerator / denominator)
    return l_nt_xent


l_perfect = nt_xent(1, -1, temp, N)
l_random = nt_xent(0, 0, temp, N)
l_worst = nt_xent(-1, 1, temp, N)

print(f"N = {N}, temp = {temp}")
print(f"Perfect: {l_perfect:.4f}")
print(f"Random: {l_random:.4f}")
print(f"Worst: {l_worst:.4f}")

# %%
