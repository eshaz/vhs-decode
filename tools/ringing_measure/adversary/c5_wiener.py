"""Claim 5: w = (lambda - bulk)/lambda == 1/(1+rho), and is `bulk` biased?"""
import sys
import numpy as np
sys.path.insert(0, "/workspaces/vhs-decode")
from vhsdecode.models import information_extrapolation as ie

rng = np.random.default_rng(31)

print("=== 5a. The algebra ===")
print("""  lambda = bulk + signal, SNR := signal/bulk.
  w = (lambda - bulk)/lambda = signal/(bulk+signal) = SNR/(1+SNR).
  With rho := 1/SNR = bulk/signal:  1/(1+rho) = 1/(1+bulk/signal)
                                             = signal/(signal+bulk) = w.
  IDENTITY CONFIRMED -- but it is a definitional restatement: it holds for
  ANY two positive numbers once rho is DEFINED as the noise-to-signal
  ratio.  It is not an independent arrival at the same law; it is the same
  MMSE shrinkage written twice.  The project's correction-gain note defines
  rho as MODEL ERROR over signal, which is only the eigenvalue bulk if all
  model error is isotropic noise -- see 5d.""")

print("\n=== 5b. Is the MP bulk's MEDIAN an unbiased estimate of the noise level? ===")
print("  For unit-norm noise rows the mean eigenvalue is exactly 1.")
print("  The code takes the MEDIAN of the sub-edge eigenvalues.")
print(f"  {'N':>4} {'L':>6} {'mean':>8} {'median':>8} {'bias':>8} {'ratio':>7}")
for N, L in ((4, 512), (8, 512), (16, 512), (32, 512), (64, 512),
             (16, 64), (16, 128), (16, 1024), (128, 512)):
    med, mean = [], []
    for _ in range(200):
        D = rng.standard_normal((N, L))
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        v = np.clip(np.linalg.eigvalsh(D @ D.T)[::-1], 0, None)
        edge = (1 + np.sqrt(N / L)) ** 2
        q = v[v <= edge]
        med.append(np.median(q) if q.size else 1.0)
        mean.append(v.mean())
    m, mu = np.mean(med), np.mean(mean)
    print(f"  {N:>4} {L:>6} {mu:>8.4f} {m:>8.4f} {m-mu:>+8.4f} {m/mu:>7.4f}")
print("""  The median of a Marchenko-Pastur bulk is BELOW its mean for every
  aspect ratio (MP is right-skewed), so `bulk` is biased LOW.
  Direction of the consequence: w = (lam - bulk)/lam is biased HIGH,
  i.e. the loop removes MORE than the Wiener optimum.  That is the
  opposite of the arc's own 'apply at half the believed optimum' law.""")

print("\n=== 5c. How biased is w itself? ===")
print("  One planted direction at a known SNR; compare the estimated w with")
print("  the true Wiener weight.")
print(f"  {'true SNR':>9} {'true w':>8} {'est w':>8} {'excess':>8}")
N, L = 16, 512
for snr in (0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 40.0):
    est = []
    for _ in range(120):
        d = rng.standard_normal(L); d /= np.linalg.norm(d)
        comps = {}
        for i in range(N):
            n = rng.standard_normal(L); n /= np.linalg.norm(n)
            comps[f"c{i}"] = {"x": n + np.sqrt(snr * N) / np.sqrt(N)
                              * rng.standard_normal() * d}
        f = ie.ellipsoid(comps, "x")
        if f["significant"]:
            lam = f["eigenvalues"][0]
            est.append((lam - f["bulk"]) / lam)
    if est:
        # true w for a spiked model: signal share of the top eigenvalue
        # measured against the true noise level of 1.0
        f = None
        est = np.mean(est)
        # true: lam = 1 + N*snr/N ... use the empirical lam with bulk=1
        print(f"  {snr:>9.2f} {snr/(1+snr):>8.4f} {est:>8.4f} "
              f"{est - snr/(1+snr):>+8.4f}")

print("\n=== 5d. When is bulk NOT the noise level at all? ===")
print("  On the tool's own nested differential matrix the sub-edge")
print("  eigenvalues are STRUCTURAL ZEROS, so bulk == 0 exactly and")
print("  w == 1 for every direction: the 'Wiener filter' degenerates to a")
print("  hard projection with no shrinkage whatsoever.")


def nested(v):
    n = len(v)
    o = {f"c{i}": {"x": v[i]} for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            o[f"c{i} vs c{j}"] = {"x": v[i] - v[j]}
    return o


for n, L in ((13, 264), (13, 119), (27, 258)):
    m = nested([rng.standard_normal(L) for _ in range(n)])
    f = ie.ellipsoid(m, "x")
    w = np.clip((f["eigenvalues"][:f["significant"]] - f["bulk"])
                / f["eigenvalues"][:f["significant"]], 0, 1)
    print(f"  {n} fields, {L} places: bulk = {f['bulk']:.3e}, "
          f"Wiener weights all in [{w.min():.6f}, {w.max():.6f}]")
print("""  So on the real decode -- the only place the method has been run on
  data -- the per-direction Wiener weighting the document credits with
  the 2-passes-and-exact-recovery result is NOT ACTIVE.  Every weight is
  1.0.  It is the hard projection, and the §3.1 comparison against the
  'flat half-step' is comparing a=1.0 with a=0.5, not Wiener with flat.""")
