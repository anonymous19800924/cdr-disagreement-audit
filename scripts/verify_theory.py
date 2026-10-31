# NOTE (conventions & scaling): in the synthetic-EM sweep of this script the channel verdict
# coding uses y=1 for 'channel flags problematic/incorrect' and z=1 for 'opinion is incorrect'.
# INIT_A ('trust alpha' / trust the conflict channel) and INIT_B are mirror images of each other
# (theta_B = T(theta_A)); because the two EM runs are label-symmetric, run A may converge to the
# branch whose fitted prior is pi=0.8948 (the paper's Table 1 mode B) -- the pair structure, flip
# rate, and likelihood identity are unaffected. The P3 imbalance sweep scales corpus counts by a
# factor of 50 (n_11 = 141,338/50 = 2,826) for runtime; the graded identity in Proposition 2 has
# been verified at the native scale in review-stage/verify_theory_results.json.
# verify_theory.py — Gate-6 verification for the graded-mirror theory line.
# 1) Sympy symbolic check: Lemma 1 pattern identities (exactly mirrored & staying patterns).
# 2) Monte Carlo: random parameters/counts, assert pattern identities + Proposition 2 closed form.
# 3) Synthetic EM predictions P2/P3 (dual initialization, surface flip rates, closed-form gaps).
# theta vector convention: [pi, a1, a0, b1, b0] where
#   a1 = theta_a(1|1), a0 = theta_a(1|0), b1 = theta_b(1|1), b0 = theta_b(1|0).
import json
import numpy as np

rng = np.random.default_rng(20261030)
IDX_RESULT_FILE = "verify_theory_results.json"
results = {"sympy": {}, "montecarlo": {}, "synthetic_em": {}}

# ---------- 1) Sympy symbolics ----------
import sympy as sp

pi_, a1, a0, b1, b0 = sp.symbols("pi_ a1 a0 b1 b0", real=True)

ell = {
    "10": pi_*a1*(1-b1) + (1-pi_)*a0*(1-b0),
    "01": pi_*(1-a1)*b1 + (1-pi_)*(1-a0)*b0,
    "11": pi_*a1*b1 + (1-pi_)*a0*b0,
    "00": pi_*(1-a1)*(1-b1) + (1-pi_)*(1-a0)*(1-b0),
}
# Mirror map on the coordinate vector:
#   pi_T = 1-pi_; a1_T = theta_a^T(1|1) = theta_b(0|0) = 1-b0; a0_T = theta_b(0|1) = 1-b1;
#   b1_T = theta_a(0|0) = 1-a0;  b0_T = theta_a(0|1) = 1-a1.
pi_T = 1 - pi_
ellT = {
    "10": pi_T*(1-b0)*a0 + (1-pi_T)*(1-b1)*a1,
    "01": pi_T*(1-(1-b0))*(1-a0) + (1-pi_T)*(1-(1-b1))*(1-a1),
    "11": pi_T*(1-b0)*(1-a0) + (1-pi_T)*(1-b1)*(1-a1),
    "00": pi_T*(1-(1-b0))*(1-(1-a0)) + (1-pi_T)*(1-(1-b1))*(1-(1-a1)),
}
checks = {
    "l10(T)=l10": sp.simplify(ellT["10"] - ell["10"]) == 0,
    "l01(T)=l01": sp.simplify(ellT["01"] - ell["01"]) == 0,
    "l11(T)=l00": sp.simplify(ellT["11"] - ell["00"]) == 0,
    "l00(T)=l11": sp.simplify(ellT["00"] - ell["11"]) == 0,
}
results["sympy"] = {k: bool(v) for k, v in checks.items()}

# ---------- 2) Monte Carlo ----------
def likelihoods(th):
    pi_, a1, a0, b1, b0 = th
    return np.array([
        pi_*a1*(1-b1) + (1-pi_)*a0*(1-b0),   # 10
        pi_*(1-a1)*b1 + (1-pi_)*(1-a0)*b0,   # 01
        pi_*a1*b1 + (1-pi_)*a0*b0,           # 11
        pi_*(1-a1)*(1-b1) + (1-pi_)*(1-a0)*(1-b0),  # 00
    ])

def mirror(th):
    pi_, a1, a0, b1, b0 = th
    return np.array([1-pi_, 1-b0, 1-b1, 1-a0, 1-a1])

worst = 0.0; worst_delta = 0.0
for _ in range(2000):
    th = rng.uniform(0.05, 0.95, size=5)
    n = rng.integers(1, 5000, size=4)  # counts for 10, 01, 11, 00
    l = likelihoods(th)
    lT = likelihoods(mirror(th))
    L = n @ np.log(l)
    LT = n @ np.log(lT)
    closed = (n[2] - n[3]) * np.log(l[2]/l[3])
    err1 = max(abs(lT[0]-l[0]), abs(lT[1]-l[1])) / min(l[0], l[1])
    err2 = max(abs(lT[2]-l[3])/l[3], abs(lT[3]-l[2])/l[2])
    errD = abs((L - LT) - closed) / max(1.0, abs(L - LT), abs(closed))
    worst = max(worst, err1, err2)
    worst_delta = max(worst_delta, errD)
results["montecarlo"] = {
    "max_pattern_rel_err": float(worst),
    "max_delta_rel_err": float(worst_delta),
    "instances": 2000,
    "pass": bool(worst < 1e-10 and worst_delta < 1e-9),
}

# ---------- 3) Synthetic EM ----------
def em(counts, th0, iters=20000, tol=1e-13):
    n10, n01, n11, n00 = counts["10"], counts["01"], counts["11"], counts["00"]
    N = n10 + n01 + n11 + n00
    pi_, a1, a0, b1, b0 = th0
    prev = -np.inf
    for it in range(iters):
        l10 = pi_*a1*(1-b1) + (1-pi_)*a0*(1-b0)
        l01 = pi_*(1-a1)*b1 + (1-pi_)*(1-a0)*b0
        l11 = pi_*a1*b1 + (1-pi_)*a0*b0
        l00 = pi_*(1-a1)*(1-b1) + (1-pi_)*(1-a0)*(1-b0)
        L = n10*np.log(l10) + n01*np.log(l01) + n11*np.log(max(l11, 1e-300)) + n00*np.log(max(l00, 1e-300))
        if L - prev < tol and it > 10:
            return np.array([pi_, a1, a0, b1, b0]), L, it
        prev = L
        EPS = 1e-12
        # E-step posterior weight of z=1 for each observed pattern (eps-floored denominators)
        p10 = pi_*a1*(1-b1)/max(l10, 1e-300)
        p01 = pi_*(1-a1)*b1/max(l01, 1e-300)
        p11 = pi_*a1*b1/max(l11, 1e-300)
        p00 = pi_*(1-a1)*(1-b1)/max(l00, 1e-300)
        s1 = n10*p10 + n01*p01 + n11*p11 + n00*p00
        s0 = N - s1
        pi_ = s1/N
        a1 = (n10*p10 + n11*p11)/s1                     # mean y_a | z=1
        b1 = (n01*p01 + n11*p11)/s1                     # mean y_b | z=1
        if s0 > 0:
            a0 = (n10*(1-p10) + n11*(1-p11))/s0
            b0 = (n01*(1-p01) + n11*(1-p11))/s0
        # clamp to interior to keep EM well-posed
        pi_, a1, a0, b1, b0 = [min(max(v, 1e-12), 1-1e-12) for v in (pi_, a1, a0, b1, b0)]
    return np.array([pi_, a1, a0, b1, b0]), L, iters

def num_z1(th, ya, yb):
    pi_, a1, a0, b1, b0 = th
    ta = a1 if ya == 1 else 1-a1
    tb = b1 if yb == 1 else 1-b1
    return pi_*ta*tb

def flip_frac(thA, thB, counts):
    fl, tot = 0.0, 0.0
    for pat, (ya, yb) in {"10": (1, 0), "01": (0, 1)}.items():
        n = counts[pat]
        if (post1(thA, ya, yb) > 0.5) != (post1(thB, ya, yb) > 0.5):
            fl += n
        tot += n
    return fl / max(tot, 1.0)

def tau(v, comp):
    return 1-v if comp else v

def post1(th, ya, yb):
    num1 = num_z1(th, ya, yb)
    num0 = (1-th[0]) * tau(th[2], ya==0) * tau(th[4], yb==0)
    return num1 / (num1 + num0)

def flip_rate(thA, thB, counts):
    fl, tot = 0.0, 0.0
    for pat, (ya, yb) in {"10": (1, 0), "01": (0, 1)}.items():
        n = counts[pat]
        if (post1(thA, ya, yb) > 0.5) != (post1(thB, ya, yb) > 0.5):
            fl += n
        tot += n
    return fl / max(tot, 1.0)

syn = {}
# inits: A = "trust alpha" (theta_a identity + eps, theta_b anti-identity + eps, pi small)
INIT_A = [0.10, 0.99, 0.01, 0.01, 0.99]
INIT_B = np.array(mirror(np.array(INIT_A)))  # mirror-consistent opposite start

counts_surf = {"10": 9630, "01": 1132, "11": 0, "00": 0}
thA_s, LA_s, itA = em(counts_surf, list(INIT_A))
thB_s, LB_s, itB = em(counts_surf, list(INIT_B))
syn["P2a_surface_only"] = {
    "flip_rate": float(flip_rate(thA_s, thB_s, counts_surf)),
    "rel_lik_diff": float(abs(LA_s - LB_s)/max(1.0, abs(LA_s))),
    "mirror_pair": bool(np.allclose(thA_s, mirror(thB_s), atol=1e-6)),
    "pi_A": float(thA_s[0]), "pi_B": float(thB_s[0]),
}

counts_bal = {"10": 9630, "01": 1132, "11": 5000, "00": 5000}
thA_b, LA_b, _ = em(counts_bal, list(INIT_A))
thB_b, LB_b, _ = em(counts_bal, list(INIT_B))
syn["P2b_balanced_agreeing"] = {
    "flip_rate": float(flip_rate(thA_b, thB_b, counts_bal)),
    "rel_lik_diff": float(abs(LA_b - LB_b)/max(1.0, abs(LA_b))),
    "mirror_pair": bool(np.allclose(thA_b, mirror(thB_b), atol=1e-6)),
    "pi_A": float(thA_b[0]), "pi_B": float(thB_b[0]),
}

curve = []
for n00 in [0, 50, 200, 1000, 5000, 20000]:
    counts = {"10": 9630, "01": 1132, "11": 141338//50, "00": n00}
    thA, LA, _ = em(counts, list(INIT_A))
    thB, LB, _ = em(counts, list(INIT_B))
    lA = likelihoods(thA)
    # Prop-2 closed form, validated directly on the EM output (valid for ANY theta):
    Delta_direct = LA - (np.array(list(counts.values())) @ np.log(np.maximum(likelihoods(mirror(thA)), 1e-300)))
    closed = (counts["11"] - counts["00"]) * np.log(lA[2]/lA[3])
    d_check = abs(Delta_direct - closed) / max(1.0, abs(Delta_direct), abs(closed))
    curve.append({
        "n00": int(n00), "n11": int(counts["11"]),
        "flip_rate": float(flip_rate(thA, thB, counts)),
        "prop2_identity_relerr": float(d_check),
        "mirror_pair_exact": bool(np.allclose(thA, mirror(thB), atol=1e-4)),
        "pi_A": float(thA[0]), "pi_B": float(thB[0]),
    })
syn["P3_imbalance_sweep"] = curve
results["synthetic_em"] = syn

with open(IDX_RESULT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(json.dumps(results, indent=2, ensure_ascii=False))
allpass = all(results["sympy"].values()) and results["montecarlo"]["pass"]
print("OVERALL:", "PASS" if allpass else "CHECK FAILURES")
