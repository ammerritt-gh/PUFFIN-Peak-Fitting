import numpy as np
import matplotlib.pyplot as plt

# --- Using the same simulation as before ---
#np.random.seed()
n_bins = 50
true_rates = np.random.uniform(0, 0.5, n_bins)
t_exp = 60.0
raw_counts = np.random.poisson(true_rates * t_exp)
norm_rates = raw_counts / t_exp
sigma_rates = np.sqrt(raw_counts) / t_exp

# Reconstruct counts
from copy import deepcopy
def reconstruct_counts_from_rates(r, sigma_r, fallback_t=None, replace_zeros_with='median'):
    r = np.asarray(r, dtype=float)
    sigma_r = np.asarray(sigma_r, dtype=float)
    t = np.full_like(r, np.nan)
    d = np.zeros_like(r)
    bad_sigma = sigma_r <= 0
    valid = (r > 0) & (~bad_sigma)
    if np.any(valid):
        t[valid] = r[valid] / (sigma_r[valid] ** 2)
        d[valid] = r[valid] * t[valid]
    zeros = (r == 0) & (~bad_sigma)
    if np.any(zeros):
        if replace_zeros_with in ('median', 'mean'):
            if np.any(valid):
                rep_t = float(np.median(t[valid])) if replace_zeros_with=='median' else float(np.mean(t[valid]))
            else:
                rep_t = fallback_t
            if rep_t is not None:
                t[zeros] = rep_t
                d[zeros] = (sigma_r[zeros]*t[zeros])**2
        elif replace_zeros_with=='fallback' and fallback_t is not None:
            t[zeros] = fallback_t
            d[zeros] = (sigma_r[zeros]*t[zeros])**2
    d[bad_sigma] = np.nan
    t[bad_sigma] = np.nan
    return d, t

d_recon, t_recon = reconstruct_counts_from_rates(norm_rates, sigma_rates, replace_zeros_with='median')

# --- Plotting ---
fig, ax1 = plt.subplots(figsize=(10, 5))

bins = np.arange(n_bins)

# Raw counts
ax1.bar(bins-0.2, raw_counts, width=0.4, label='Raw counts (d)', color='C0', alpha=0.6)

# Reconstructed counts
ax1.bar(bins+0.2, d_recon, width=0.4, label='Reconstructed counts', color='C1', alpha=0.6)

ax1.set_xlabel('Bin')
ax1.set_ylabel('Counts')
ax1.set_title('Raw vs Reconstructed Counts from Normalized Rates')

# Secondary axis: normalized rates
ax2 = ax1.twinx()
ax2.errorbar(bins, norm_rates, yerr=sigma_rates, fmt='o', color='C2', label='Normalized rates (r)')
ax2.set_ylabel('Normalized rate (counts/sec)')

# Legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()
