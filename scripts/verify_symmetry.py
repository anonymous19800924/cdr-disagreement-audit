# verify_symmetry_group.py — checks for the V4 symmetry refinement and identifiability threshold
import numpy as np
import sympy as sp

# ---------- 1) symbolic group algebra ----------
pi_, a1, a0, b1, b0 = sp.symbols("pi_ a1 a0 b1 b0", real=True)
th = {"pi": pi_, "a1": a1, "a0": a0, "b1": b1, "b0": b0}

def S(t):   # label swap: pi -> 1-pi; theta_j(y|z) -> theta_j(y|1-z)
    return {"pi": 1 - t["pi"], "a1": t["a0"], "a0": t["a1"], "b1": t["b0"], "b0": t["b1"]}

def T(t):   # mirror: pi -> 1-pi; theta_a^T(y|z)=theta_b(1-y|1-z); theta_b^T(y|z)=theta_a(1-y|1-z)
    return {"pi": 1 - t["pi"],
            "a1": 1 - t["b0"],   # theta_a^T(1|1) = theta_b(0|0)
            "a0": 1 - t["b1"],   # theta_a^T(1|0) = theta_b(0|1)
            "b1": 1 - t["a0"],   # theta_b^T(1|1) = theta_a(0|0)
            "b0": 1 - t["a1"]}   # theta_b^T(1|0) = theta_a(0|1)

def U(t):   # channel-verdict exchange U = S o T
    return {"pi": t["pi"],
            "a1": 1 - t["b1"],   # theta_a^U(1|1) = theta_b(0|1)
            "a0": 1 - t["b0"],   # theta_a^U(1|0) = theta_b(0|0)
            "b1": 1 - t["a1"],
            "b0": 1 - t["a0"]}

def key_equal(t1, t2):
    return all(sp.simplify(t1[k] - t2[k]) == 0 for k in t1)

checks = {}
checks["S^2=id"] = key_equal(S(S(th)), th)
checks["T^2=id"] = key_equal(T(T(th)), th)
checks["U^2=id"] = key_equal(U(U(th)), th)
checks["ST=TS"] = key_equal(S(T(th)), T(S(th)))
checks["S(T)=U"] = key_equal(S(T(th)), U(th))

# ---------- likelihood and posterior ----------
def lpost(t, ya, yb):
    l10 = t["pi"]*t["a1"]*(1-t["b1"]) + (1-t["pi"])*t["a0"]*(1-t["b0"])
    l01 = t["pi"]*(1-t["a1"])*t["b1"] + (1-t["pi"])*(1-t["a0"])*t["b0"]
    p10 = sp.simplify((t["pi"]*t["a1"]*(1-t["b1"])) / l10) if ya == 1 else None
    p01 = sp.simplify((t["pi"]*(1-t["a1"])*t["b1"]) / l01) if ya == 0 else None
    return (l10 if ya == 1 else l01), (p10 if ya == 1 else p01)

l10, p10 = lpost(th, 1, 0)
l01, p01 = lpost(th, 0, 1)
l10S, p10S = lpost(S(th), 1, 0)
l10T, p10T = lpost(T(th), 1, 0)
l10U, p10U = lpost(U(th), 1, 0)
l01U, p01U = lpost(U(th), 0, 1)

checks["S: l10 invariant"] = sp.simplify(l10S - l10) == 0
checks["T: l10 invariant"] = sp.simplify(l10T - l10) == 0
checks["U: l10 invariant"] = sp.simplify(l10U - l10) == 0
checks["U: l01 invariant"] = sp.simplify(l01U - l01) == 0
checks["S: posterior flips"] = sp.simplify(p10S - (1 - p10)) == 0
checks["T: posterior flips"] = sp.simplify(p10T - (1 - p10)) == 0
checks["U: posterior preserved"] = sp.simplify(p10U - p10) == 0

# ---------- 2) generic rank of the k=3 DS model (full 8-cell data) ----------
# params: pi, a1,a0 (channel a), b1,b0, c1,c0 ; cells: all 8 patterns -> 7 free
def cells(th):
    pi_, a1, a0, b1, b0, c1, c0 = th
    out = []
    for ya in (1, 0):
        for yb in (1, 0):
            for yc in (1, 0):
                ta = a1 if ya else 1-a1; tb = b1 if yb else 1-b1; tc = c1 if yc else 1-c0 if False else (c1 if yc else 1-c1)
                # correct complements:
                ta = a1 if ya == 1 else 1-a1
                tb = b1 if yb == 1 else 1-b1
                tc = c1 if yc == 1 else 1-c1
                ta0 = a0 if ya == 1 else 1-a0
                tb0 = b0 if yb == 1 else 1-b0
                tc0 = c0 if yc == 1 else 1-c0
                out.append(pi_*ta*tb*tc + (1-pi_)*ta0*tb0*tc0)
    return out

syms = sp.symbols("pi_ a1 a0 b1 b0 c1 c0", real=True)
C = cells(syms)
J = sp.Matrix([[sp.diff(c, s) for s in syms] for c in C[:-1]])  # drop one redundant cell
pt = {syms[0]: sp.Rational(1, 3), syms[1]: sp.Rational(8, 10), syms[2]: sp.Rational(2, 10),
      syms[3]: sp.Rational(7, 10), syms[4]: sp.Rational(3, 10), syms[5]: sp.Rational(6, 10),
      syms[6]: sp.Rational(4, 10)}
r = J.subs(pt).rank()
print("k=2 group checks:", checks)
print("k=3 Jacobian (7x7) rank at generic interior point:", r)

# ---------- 3) k=2 full-data fiber dimension at a generic point ----------
syms2 = sp.symbols("pi_ a1 a0 b1 b0", real=True)
def cells2(th):
    pi_, a1, a0, b1, b0 = th
    out = []
    for ya in (1, 0):
        for yb in (1, 0):
            ta = a1 if ya == 1 else 1-a1
            tb = b1 if yb == 1 else 1-b1
            ta0 = a0 if ya == 1 else 1-a0
            tb0 = b0 if yb == 1 else 1-b0
            out.append(pi_*ta*tb + (1-pi_)*ta0*tb0)
    return out
C2 = cells2(syms2)
J2 = sp.Matrix([[sp.diff(c, s) for s in syms2] for c in C2[:-1]])  # 3 x 5
pt2 = {syms2[0]: sp.Rational(1, 3), syms2[1]: sp.Rational(8, 10), syms2[2]: sp.Rational(2, 10),
       syms2[3]: sp.Rational(7, 10), syms2[4]: sp.Rational(3, 10)}
r2 = J2.subs(pt2).rank()
print("k=2 full-data Jacobian (3x5) rank at generic point:", r2, "=> fiber dim =", 5 - r2)
print("ALL SYMBOLIC CHECKS:", all(checks.values()))
