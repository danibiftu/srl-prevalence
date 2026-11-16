"""
Cut-off-free prevalence mapping with a Bayesian location-scale spatial rank likelihood.

Code for: Bekalo, D.B.; Esenogho, E. Cut-Off-Free Prevalence Mapping with a Bayesian
Location-Scale Spatial Rank Likelihood. Stats (under review).

The model links an outcome to latent Gaussian variables through an unknown monotone
transformation (extended rank likelihood), with BYM2 area effects, survey cluster effects
and a log-linear model for the latent spread. Area prevalence at any cut-off, including
individual-specific cut-offs, is obtained from one fit through a clustered Bayesian bootstrap.

This single file contains
  Part 1  the method: area graph and BYM2 prior, Gibbs sampler, cut-off-free prevalence,
          comparator models (Gaussian, binary/ordinal probit BYM2), convergence diagnostics,
          data simulation;
  Part 2  the simulation study (Section 5);
  Part 3  the application to women's body mass index in Ethiopia (Section 6).

Requirements: Python >= 3.10, numpy, scipy, pandas, joblib, matplotlib; for the application
also pyreadstat, geopandas, libpysal and shapely.

Data: the DHS survey data cannot be redistributed. Register with the DHS Program
(https://dhsprogram.com) and place ETIR8AFL.dta and ETGE8AFL.shp in data/dhs/. The OCHA
administrative boundaries (Humanitarian Data Exchange) go in data/admin/ and the WHO 2007
BMI-for-age reference (bfawho2007.txt, from WHO AnthroPlus) in data/who2007/. The simulation
needs only data/W_regions.npy, data/W_zones.npy and data/zones_lookup.csv.

Licence: MIT.
"""
import argparse
import glob
import os
import pickle
import time
import zlib

import numpy as np
import pandas as pd
from scipy import optimize, special, stats
from scipy.sparse import csgraph, csr_matrix


class AreaGraph:
    """Adjacency structure of the areas and everything derived from it.

    Parameters
    ----------
    W : array (m, m)
        Symmetric 0/1 adjacency matrix (w_ii' = 1 if areas share a boundary).
    """

    def __init__(self, W):
        W = np.asarray(W, dtype=float)
        if W.shape[0] != W.shape[1] or not np.allclose(W, W.T):
            raise ValueError("W must be a symmetric square matrix")
        np.fill_diagonal(W, 0.0)
        self.W = W
        self.m = W.shape[0]
        self.degree = W.sum(axis=1)
        self.Q = np.diag(self.degree) - W

        n_comp, labels = csgraph.connected_components(csr_matrix(W), directed=False)
        self.n_components = n_comp
        self.component = labels
        self.isolated = self.degree == 0

        self.scale = np.ones(self.m)
        Q_scaled = np.zeros_like(self.Q)
        Sigma = np.zeros_like(self.Q)
        for c in range(n_comp):
            idx = np.where(labels == c)[0]
            if len(idx) == 1:
                continue
            Qc = self.Q[np.ix_(idx, idx)]
            lam, vec = np.linalg.eigh(Qc)
            keep = lam > 1e-8 * lam.max()
            Sc = (vec[:, keep] / lam[keep]) @ vec[:, keep].T
            s = np.exp(np.mean(np.log(np.diag(Sc))))
            self.scale[idx] = s
            Q_scaled[np.ix_(idx, idx)] = s * Qc
            Sigma[np.ix_(idx, idx)] = Sc / s
        self.Q_scaled = Q_scaled
        self.Sigma_scaled = Sigma

        rows = []
        for c in range(n_comp):
            idx = np.where(labels == c)[0]
            if len(idx) == 1:
                r = np.zeros(self.m)
                r[idx[0]] = 1.0
            else:
                r = np.zeros(self.m)
                r[idx] = 1.0
            rows.append(r)
        self.constraints = np.vstack(rows)
        self.n_free_u = self.m - self.constraints.shape[0]

        self.eigen = np.clip(np.linalg.eigvalsh(self.Sigma_scaled), 0.0, None)

    def kld(self, rho):
        """Kullback-Leibler divergence of BYM2(rho) from rho = 0 (eq. kld)."""
        rho = np.atleast_1d(rho)[:, None]
        g = self.eigen[None, :] - 1.0
        return 0.5 * np.sum(rho * g - np.log1p(rho * g), axis=1)

    def delta(self, rho):
        return np.sqrt(2.0 * np.maximum(self.kld(rho), 0.0))

    def ddelta(self, rho):
        rho = np.atleast_1d(rho)
        g = self.eigen[None, :] - 1.0
        dkld = 0.5 * np.sum(rho[:, None] * g ** 2 / (1.0 + rho[:, None] * g), axis=1)
        return dkld / np.maximum(self.delta(rho), 1e-300)

    def pc_rho_setup(self, U=0.5, alpha=2.0 / 3.0):
        """Choose theta so that P(rho < U) = alpha (truncated if delta(1) is finite)."""
        d_U = self.delta(U)[0]
        d_1 = self.delta(1.0 - 1e-12)[0]

        def prob(theta):
            num = -np.expm1(-theta * d_U)
            den = -np.expm1(-theta * d_1) if np.isfinite(d_1) else 1.0
            return num / den - alpha

        self.pc_theta = optimize.brentq(prob, 1e-8, 1e4)
        return self.pc_theta

    def log_pc_rho(self, rho):
        """Log PC prior density of rho, up to a constant."""
        th = self.pc_theta
        return np.log(th) - th * self.delta(rho) + np.log(np.abs(self.ddelta(rho)))

    def sample_icar(self, rng):
        """Draw the scaled structured effect u from its constrained prior."""
        vals, vecs = np.linalg.eigh(self.Sigma_scaled)
        keep = vals > 1e-10
        return vecs[:, keep] @ (np.sqrt(vals[keep]) * rng.standard_normal(keep.sum()))


def grid_adjacency(nrow, ncol):
    """Rook adjacency matrix for an nrow x ncol lattice (areas numbered row-wise)."""
    m = nrow * ncol
    W = np.zeros((m, m))
    for r in range(nrow):
        for c in range(ncol):
            i = r * ncol + c
            if c + 1 < ncol:
                W[i, i + 1] = W[i + 1, i] = 1
            if r + 1 < nrow:
                W[i, i + ncol] = W[i + ncol, i] = 1
    return W


def adjacency_from_geodataframe(gdf):
    """Queen-contiguity adjacency matrix from a GeoDataFrame of area polygons."""
    from libpysal.weights import Queen

    w = Queen.from_dataframe(gdf, use_index=False)
    W = np.zeros((len(gdf), len(gdf)))
    for i, nbrs in w.neighbors.items():
        W[i, nbrs] = 1
    return W


def rtruncnorm(mu, lo, hi, rng):
    """Vectorised draws from N(mu, 1) truncated to (lo, hi).

    Uses the inverse-CDF method on the tail closer to the mean, which stays
    accurate when the interval lies far in a tail.
    """
    mu = np.asarray(mu, dtype=float)
    a = np.asarray(lo, dtype=float) - mu
    b = np.asarray(hi, dtype=float) - mu
    flip = a > 0
    a2 = np.where(flip, -b, a)
    b2 = np.where(flip, -a, b)
    pa = special.ndtr(a2)
    pb = special.ndtr(b2)
    u = rng.random(mu.shape)
    p = pa + u * (pb - pa)
    x = special.ndtri(np.clip(p, 1e-300, 1 - 1e-16))
    bad = ~np.isfinite(x) | (pb - pa <= 0)
    if np.any(bad):
        x = np.where(bad, np.where(np.isfinite(b2), b2 - 1e-9, a2 + 1e-9), x)
    x = np.where(flip, -x, x)
    eps = 1e-12
    with np.errstate(invalid="ignore"):
        lo_g = np.where(np.isfinite(a), a + eps * (1 + np.abs(a)), -np.inf)
        hi_g = np.where(np.isfinite(b), b - eps * (1 + np.abs(b)), np.inf)
    x = np.minimum(np.maximum(x, lo_g), hi_g)
    return mu + x


class SpatialRankLikelihood:
    """Bayesian spatial rank-likelihood model with BYM2 area and cluster effects.

    Parameters
    ----------
    y : array (N,)
        Outcome, ordinal (any ordered numeric coding) or continuous.
    X : array (N, p)
        Covariates without an intercept column (continuous ones standardised).
    area : int array (N,)
        Area index 0..m-1 of each individual.
    cluster : int array (N,) or None
        Cluster index 0..K-1 of each individual (clusters nested in areas).
    W : array (m, m)
        Adjacency matrix of the areas.
    Z_scale : array (N, q) or None
        Optional design for heteroscedastic latent errors, log sd_a = Z_a' gamma,
        without an intercept (a reference group has sd 1). None gives the
        homoscedastic model.
    """

    def __init__(self, y, X, area, cluster, W, beta_sd=10.0,
                 pc_sigma_phi=(1.0, 0.01), pc_sigma_eta=(1.0, 0.01),
                 pc_rho=(0.5, 2.0 / 3.0), Z_scale=None, gamma_sd=1.0):
        self.y = np.asarray(y, dtype=float)
        X = np.asarray(X, dtype=float)
        self.X_mean = X.mean(axis=0)
        self.X = X - self.X_mean
        self.N, self.p = self.X.shape
        self.area = np.asarray(area, dtype=int)
        self.graph = AreaGraph(W)
        self.m = self.graph.m
        self.graph.pc_rho_setup(*pc_rho)
        self.has_cluster = cluster is not None
        self.cluster = np.asarray(cluster, dtype=int) if self.has_cluster else np.zeros(self.N, int)
        self.K = self.cluster.max() + 1

        if np.any(np.all(self.X == self.X[0], axis=0)):
            raise ValueError("X must not contain a constant column (no intercept)")

        self.beta_prec = 1.0 / beta_sd ** 2
        self.lam_phi = -np.log(pc_sigma_phi[1]) / pc_sigma_phi[0]
        self.lam_eta = -np.log(pc_sigma_eta[1]) / pc_sigma_eta[0]

        self.levels_val, self.lev = np.unique(self.y, return_inverse=True)
        self.L = len(self.levels_val)
        if self.L < 2:
            raise ValueError("y must take at least two distinct values")
        self.order = np.argsort(self.lev, kind="stable")
        counts = np.bincount(self.lev, minlength=self.L)
        self.starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        self.odd = (self.lev % 2) == 1
        self.even = ~self.odd

        self.XtX = self.X.T @ self.X
        P = self.XtX + self.beta_prec * np.eye(self.p)
        self.P_beta_chol = np.linalg.cholesky(P)
        self.n_area = np.bincount(self.area, minlength=self.m).astype(float)
        self.n_clus = np.bincount(self.cluster, minlength=self.K).astype(float)
        self.hetero = Z_scale is not None
        self.Zs = np.asarray(Z_scale, dtype=float) if self.hetero else np.zeros((self.N, 0))
        self.q = self.Zs.shape[1]
        self.gamma_prec = 1.0 / gamma_sd ** 2
        self.D_free = (self.N + self.p + self.graph.n_free_u + self.m
                       + (self.K if self.has_cluster else 0) + self._n_extra_free())

    def _level_bounds(self, z):
        zs = z[self.order]
        lev_max = np.maximum.reduceat(zs, self.starts)
        lev_min = np.minimum.reduceat(zs, self.starts)
        lo = np.concatenate([[-np.inf], lev_max])[self.lev]
        hi = np.concatenate([lev_min[1:], [np.inf]])[self.lev]
        return lo, hi

    def _update_z(self, z, mu, rng, sd=None):
        for block in (self.odd, self.even):
            lo, hi = self._level_bounds(z)
            if sd is None:
                z[block] = rtruncnorm(mu[block], lo[block], hi[block], rng)
            else:
                m_, s_ = mu[block], sd[block]
                z[block] = m_ + s_ * rtruncnorm(np.zeros(block.sum()), (lo[block] - m_) / s_,
                                                (hi[block] - m_) / s_, rng)
        return z

    def _sd(self, s):
        """Latent error standard deviations (ones in the homoscedastic model)."""
        return np.exp(self.Zs @ s["gamma"]) if self.hetero else np.ones(self.N)

    def _n_extra_free(self):
        """Extra coordinates rescaled by the scale move (e.g. thresholds)."""
        return 0

    def _update_latent(self, s, rng, prop_sd, acc):
        """Latent update; the rank-likelihood version uses only the ranks of y."""
        return self._update_z(s["z"], self._mu(s), rng, self._sd(s) if self.hetero else None)

    def _scale_keys(self):
        return ("z", "beta", "u", "v", "eta")

    def _shift_keys(self):
        return ("z",)

    def _init_prop_sd(self):
        d = {"rho": 0.5, "eta": 0.2, "phi": 0.2}
        if getattr(self, "hetero", False):
            d["gamma"] = np.full(self.q, 0.05)
        return d

    def _extra_store(self, s):
        """Extra per-draw quantities to keep (dict of name -> value)."""
        return {"gamma": s["gamma"].copy()} if self.hetero else {}

    def _phi(self, s):
        return s["sigma_phi"] * (np.sqrt(s["rho"]) * s["u"] + np.sqrt(1 - s["rho"]) * s["v"])

    def _marg_ll(self, sig, rho, S, nA=None):
        """Log density of the area totals S with the area effects integrated out.

        S_i = sum_a r_a / sd_a^2 and nA_i = sum_a 1 / sd_a^2 (plain sums and counts
        in the homoscedastic model), so that S_i | phi_i ~ N(nA_i phi_i, nA_i).
        """
        nA = self.n_area if nA is None else nA
        keep = self.n_area > 0
        n = nA[keep]
        Cphi = sig * sig * ((1 - rho) * np.eye(self.m) + rho * self.graph.Sigma_scaled)[np.ix_(keep, keep)]
        V = n[:, None] * Cphi * n[None, :] + np.diag(n)
        L = np.linalg.cholesky(V)
        a = np.linalg.solve(L, S[keep])
        return -np.sum(np.log(np.diag(L))) - 0.5 * a @ a

    def _mu(self, s):
        return self.X @ s["beta"] + self._phi(s)[self.area] + s["eta"][self.cluster]

    def _init_state(self, rng, disperse):
        ranks = stats.rankdata(self.y, method="average")
        z = special.ndtri(ranks / (self.N + 1))
        s = dict(beta=np.zeros(self.p), u=np.zeros(self.m), v=np.zeros(self.m),
                 eta=np.zeros(self.K), sigma_phi=0.5, rho=0.5,
                 sigma_eta=0.5 if self.has_cluster else 0.0, z=z, gamma=np.zeros(self.q))
        if disperse:
            s["beta"] = rng.normal(0, 0.5, self.p)
            s["sigma_phi"] = rng.uniform(0.2, 1.2)
            s["rho"] = rng.uniform(0.1, 0.9)
            if self.has_cluster:
                s["sigma_eta"] = rng.uniform(0.1, 0.8)
            s["z"] = z * rng.uniform(0.7, 1.4)
        return s

    def _step(self, s, rng, group_moves, prop_sd):
        X, z = self.X, s["z"]
        acc = {}

        z = self._update_latent(s, rng, prop_sd, acc)
        s["z"] = z

        lam = 1.0 / self._sd(s) ** 2 if self.hetero else None
        nA = np.bincount(self.area, weights=lam, minlength=self.m) if self.hetero else self.n_area

        phi = self._phi(s)
        r = z - phi[self.area] - s["eta"][self.cluster]
        if self.hetero:
            rhs = X.T @ (lam * r)
            Lc = np.linalg.cholesky((X * lam[:, None]).T @ X + self.beta_prec * np.eye(self.p))
        else:
            rhs = X.T @ r
            Lc = self.P_beta_chol
        mean = np.linalg.solve(Lc.T, np.linalg.solve(Lc, rhs))
        s["beta"] = mean + np.linalg.solve(Lc.T, rng.standard_normal(self.p))
        xb = X @ s["beta"]

        r = z - xb - s["eta"][self.cluster]
        S = np.bincount(self.area, weights=r * lam if self.hetero else r, minlength=self.m)
        cur = self._marg_ll(s["sigma_phi"], s["rho"], S, nA)

        prop = s["sigma_phi"] * np.exp(prop_sd["phi"] * rng.standard_normal())
        new = self._marg_ll(prop, s["rho"], S, nA)
        log_r = (new - cur) - self.lam_phi * (prop - s["sigma_phi"]) + np.log(prop / s["sigma_phi"])
        ok = np.log(rng.random()) < log_r
        if ok:
            s["sigma_phi"], cur = prop, new
        acc["phi"] = ok

        lg = np.log(s["rho"]) - np.log1p(-s["rho"]) + prop_sd["rho"] * rng.standard_normal()
        prop = 1.0 / (1.0 + np.exp(-lg))
        if 1e-8 < prop < 1 - 1e-8:
            new = self._marg_ll(s["sigma_phi"], prop, S, nA)
            log_r = (new - cur) + (self.graph.log_pc_rho(prop)[0] - self.graph.log_pc_rho(s["rho"])[0]) \
                + (np.log(prop) + np.log1p(-prop)) - (np.log(s["rho"]) + np.log1p(-s["rho"]))
            ok = np.log(rng.random()) < log_r
        else:
            ok = False
        if ok:
            s["rho"] = prop
        acc["rho"] = ok

        au = s["sigma_phi"] * np.sqrt(s["rho"])
        av = s["sigma_phi"] * np.sqrt(1 - s["rho"])
        m = self.m
        Lam = np.zeros((2 * m, 2 * m))
        Lam[:m, :m] = self.graph.Q_scaled + np.diag(au * au * nA) + 1e-9 * np.eye(m)
        Lam[m:, m:] = np.eye(m) + np.diag(av * av * nA)
        Lam[:m, m:] = Lam[m:, :m] = np.diag(au * av * nA)
        b = np.concatenate([au * S, av * S])
        Lw = np.linalg.cholesky(Lam)
        mean = np.linalg.solve(Lw.T, np.linalg.solve(Lw, b))
        w = mean + np.linalg.solve(Lw.T, rng.standard_normal(2 * m))
        C = np.hstack([self.graph.constraints, np.zeros_like(self.graph.constraints)])
        LinvCt = np.linalg.solve(Lw.T, np.linalg.solve(Lw, C.T))
        w = w - LinvCt @ np.linalg.solve(C @ LinvCt, C @ w)
        s["u"], s["v"] = w[:m], w[m:]

        if self.has_cluster:
            phi = self._phi(s)
            r = z - xb - phi[self.area]
            Sk = np.bincount(self.cluster, weights=r * lam if self.hetero else r, minlength=self.K)
            nK = np.bincount(self.cluster, weights=lam, minlength=self.K) if self.hetero else self.n_clus

            def lp_eta(sig):
                v = nK + nK ** 2 * sig ** 2
                return -0.5 * np.sum(np.log(v) + Sk ** 2 / v) - self.lam_eta * sig + np.log(sig)

            prop = s["sigma_eta"] * np.exp(prop_sd["eta"] * rng.standard_normal())
            ok = np.log(rng.random()) < lp_eta(prop) - lp_eta(s["sigma_eta"])
            if ok:
                s["sigma_eta"] = prop
            acc["eta"] = ok

            prec = nK + 1.0 / s["sigma_eta"] ** 2
            s["eta"] = Sk / prec + rng.standard_normal(self.K) / np.sqrt(prec)

        if self.hetero:
            e2 = (z - self._mu(s)) ** 2
            lsd = self.Zs @ s["gamma"]

            def ll_g(lsd_, g_):
                return -np.sum(lsd_) - 0.5 * np.sum(e2 * np.exp(-2 * lsd_)) - 0.5 * self.gamma_prec * g_ @ g_

            cur = ll_g(lsd, s["gamma"])
            ok_all = []
            for k in range(self.q):
                g_new = s["gamma"].copy()
                g_new[k] += prop_sd["gamma"][k] * rng.standard_normal()
                lsd_new = lsd + self.Zs[:, k] * (g_new[k] - s["gamma"][k])
                new = ll_g(lsd_new, g_new)
                ok = np.log(rng.random()) < new - cur
                if ok:
                    s["gamma"], lsd, cur = g_new, lsd_new, new
                ok_all.append(ok)
            acc["gamma"] = np.array(ok_all)
            lam = np.exp(-2 * lsd)

        if group_moves:
            e = z - self._mu(s)
            T = ((e @ e if not self.hetero else np.sum(lam * e * e)) + self.beta_prec * s["beta"] @ s["beta"]
                 + s["u"] @ self.graph.Q_scaled @ s["u"] + s["v"] @ s["v"])
            if self.has_cluster:
                T += np.sum(s["eta"] ** 2) / s["sigma_eta"] ** 2
            bsc = np.sqrt(rng.gamma(self.D_free / 2.0, 2.0 / T))
            for key in self._scale_keys():
                s[key] = s[key] * bsc
            delta = rng.normal(-s["v"].mean(), 1.0 / np.sqrt(self.m))
            s["v"] = s["v"] + delta
            kappa = s["sigma_phi"] * np.sqrt(1 - s["rho"]) * delta
            for key in self._shift_keys():
                s[key] = s[key] + kappa
        return s, acc

    def fit(self, n_iter=3000, burn=1000, thin=1, n_chains=4, seed=0,
            group_moves=True, store_z=False, verbose=False):
        """Run n_chains chains; returns a dict of arrays with shape (chain, draw, ...)."""
        ss = np.random.SeedSequence(seed)
        n_keep = (n_iter - burn) // thin
        out = {k: np.empty((n_chains, n_keep) + shp) for k, shp in
               [("beta", (self.p,)), ("phi", (self.m,)), ("eta", (self.K,)),
                ("sigma_phi", ()), ("rho", ()), ("sigma_eta", ())]}
        if store_z:
            out["z"] = np.empty((n_chains, n_keep, self.N), dtype=np.float32)
        acc_rate = {k: [] for k in self._init_prop_sd()}

        for ch, child in enumerate(ss.spawn(n_chains)):
            rng = np.random.default_rng(child)
            s = self._init_state(rng, disperse=ch > 0)
            prop_sd = self._init_prop_sd()
            hist = {k: [] for k in prop_sd}
            j = 0
            for it in range(n_iter):
                s, acc = self._step(s, rng, group_moves, prop_sd)
                for k, v in acc.items():
                    hist[k].append(v)
                if it < burn and (it + 1) % 50 == 0:
                    for k in prop_sd:
                        if hist[k]:
                            rate = np.mean(np.asarray(hist[k][-50:], dtype=float), axis=0)
                            prop_sd[k] = prop_sd[k] * np.exp(rate - 0.40)
                if it >= burn and (it - burn) % thin == 0 and j < n_keep:
                    phi = self._phi(s)
                    out["beta"][ch, j] = s["beta"]
                    out["phi"][ch, j] = phi - phi.mean()
                    out["eta"][ch, j] = s["eta"]
                    out["sigma_phi"][ch, j] = s["sigma_phi"]
                    out["rho"][ch, j] = s["rho"]
                    out["sigma_eta"][ch, j] = s["sigma_eta"]
                    if store_z:
                        out["z"][ch, j] = s["z"]
                    for k, val in self._extra_store(s).items():
                        if k not in out:
                            out[k] = np.empty((n_chains, n_keep) + np.shape(val))
                        out[k][ch, j] = val
                    j += 1
            for k in acc_rate:
                if hist[k]:
                    acc_rate[k].append(np.round(np.mean(np.asarray(hist[k][burn:], dtype=float), axis=0), 3).tolist())
            if verbose:
                print(f"chain {ch + 1}/{n_chains} done")
        out["acceptance"] = acc_rate
        self.draws = out
        return out

    def latent_means(self, draws=None, flat=True):
        """Latent means mu_a for each retained draw (centred area effects).

        Centring the area effects shifts every mu_a by the same constant, which
        is absorbed by g and does not affect ranks, thresholds or prevalence.
        """
        d = self.draws if draws is None else draws
        beta = d["beta"].reshape(-1, self.p)
        phi = d["phi"].reshape(-1, self.m)
        eta = d["eta"].reshape(-1, self.K)
        return beta @ self.X.T + phi[:, self.area] + eta[:, self.cluster]

    def latent_sds(self, draws=None):
        """Latent error standard deviations for each retained draw (ones if homoscedastic)."""
        d = self.draws if draws is None else draws
        if not self.hetero:
            return None
        return np.exp(d["gamma"].reshape(-1, self.q) @ self.Zs.T)


def latent_threshold(c_grid, y, mu, omega, n_newton=30, sd=None):
    """Solve sum_a omega_a Phi(t - mu_a) = sum_a omega_a 1{y_a < c} for each c in c_grid.

    Strict inequality matches the prevalence of y < c when y is recorded with ties.
    With heteroscedastic latent errors, Phi(t - mu_a) becomes Phi((t - mu_a) / sd_a).
    """
    sd = np.ones_like(mu) if sd is None else sd
    ys = np.sort(y)
    cw = np.concatenate([[0.0], np.cumsum(omega[np.argsort(y)])])
    target = cw[np.searchsorted(ys, c_grid, side="left")]
    target = np.clip(target, 1e-10, 1 - 1e-10)
    lo = np.full(len(c_grid), mu.min() - 10.0)
    hi = np.full(len(c_grid), mu.max() + 10.0)
    t = special.ndtri(target) * np.sum(omega * sd) + np.sum(omega * mu)
    t = np.clip(t, lo, hi)
    for _ in range(n_newton):
        d = (t[:, None] - mu[None, :]) / sd[None, :]
        F = special.ndtr(d) @ omega
        if np.max(np.abs(F - target)) < 1e-12:
            break
        f = (np.exp(-0.5 * d * d) / np.sqrt(2 * np.pi) / sd[None, :]) @ omega
        lo = np.where(F < target, t, lo)
        hi = np.where(F > target, t, hi)
        t_new = t - (F - target) / np.maximum(f, 1e-300)
        out = (t_new < lo) | (t_new > hi) | ~np.isfinite(t_new)
        t = np.where(out, 0.5 * (lo + hi), t_new)
    return t


def bootstrap_weights(rng, N, clusters=None):
    """Bayesian-bootstrap weights: Dirichlet(1,...,1) over individuals, or over clusters
    with each cluster's weight shared equally by its members (clustered data)."""
    if clusters is None:
        return rng.dirichlet(np.ones(N))
    nk = np.bincount(clusters)
    wk = rng.dirichlet(np.ones(len(nk)))
    return wk[clusters] / nk[clusters]


def prevalence(model, cutoffs, weights=None, n_draws=None, grid_size=64, seed=1,
               below=True, bootstrap="cluster"):
    """Posterior draws of area prevalence at (possibly individual-specific) cut-offs.

    Parameters
    ----------
    model : fitted SpatialRankLikelihood (continuous outcome)
    cutoffs : float or array (N,)
        Cut-off for each individual (a scalar applies to everyone).
    weights : array (N,) or None
        Survey weights used in the area averages (equal weights if None).
    n_draws : int or None
        Number of posterior draws to use (thinned evenly); all if None.
    below : bool
        Prevalence of y < c (True) or y >= c (False).
    bootstrap : "cluster", "individual" or "none"
        Bayesian bootstrap over clusters (default; respects the survey design), over
        individuals, or equal weights without bootstrap.

    Returns
    -------
    array (n_draws, m) of area prevalences, and array (n_draws,) of the overall prevalence.
    """
    y = model.y
    N, m = model.N, model.m
    c = np.broadcast_to(np.asarray(cutoffs, dtype=float), (N,)).copy()
    w = np.ones(N) if weights is None else np.asarray(weights, dtype=float)
    mu_all = model.latent_means()
    sd_all = model.latent_sds() if getattr(model, "hetero", False) else None
    S = mu_all.shape[0]
    idx = np.arange(S) if n_draws is None else np.linspace(0, S - 1, n_draws).astype(int)
    rng = np.random.default_rng(seed)

    uc = np.unique(c)
    grid = uc if len(uc) <= grid_size else np.quantile(c, np.linspace(0, 1, grid_size))
    area_w = np.bincount(model.area, weights=w, minlength=m)

    prev = np.empty((len(idx), m))
    overall = np.empty(len(idx))
    for j, s in enumerate(idx):
        mu = mu_all[s]
        sd = np.ones(N) if sd_all is None else sd_all[s]
        if bootstrap == "none":
            omega = np.full(N, 1.0 / N)
        else:
            omega = bootstrap_weights(rng, N, model.cluster if bootstrap == "cluster" else None)
        tau_grid = latent_threshold(grid, y, mu, omega, sd=sd)
        tau = tau_grid[np.searchsorted(grid, c)] if len(uc) <= grid_size else np.interp(c, grid, tau_grid)
        p = special.ndtr((tau - mu) / sd)
        if not below:
            p = 1.0 - p
        prev[j] = np.bincount(model.area, weights=w * p, minlength=m) / area_w
        overall[j] = np.sum(w * p) / np.sum(w)
    return prev, overall


def _log_prob_interval(a, b):
    """log{Phi(b) - Phi(a)}, computed on the tail that avoids cancellation."""
    with np.errstate(divide="ignore"):
        upper = a > 0
        p = np.where(upper, special.ndtr(-a) - special.ndtr(-b), special.ndtr(b) - special.ndtr(a))
        return np.log(np.maximum(p, 1e-300))


class OrdinalProbitBYM2(SpatialRankLikelihood):
    """Cumulative probit BYM2 model with free thresholds and a flat threshold prior.

    The thresholds are updated jointly by a random-walk Metropolis step with the
    latent variables integrated out (Cowles, 1996), and then z is drawn given
    them. The scale and location moves also rescale and shift the thresholds.
    """

    def _n_extra_free(self):
        return self.L - 1

    def _init_prop_sd(self):
        d = super()._init_prop_sd()
        d["thr"] = 0.02
        return d

    def _init_state(self, rng, disperse):
        s = super()._init_state(rng, disperse)
        z = s["z"]
        thr = np.array([0.5 * (z[self.lev == l - 1].max() + z[self.lev == l].min())
                        for l in range(1, self.L)])
        s["thr"] = thr
        return s

    def _bounds(self, thr):
        ext = np.concatenate([[-np.inf], thr, [np.inf]])
        return ext[self.lev], ext[self.lev + 1]

    def _update_latent(self, s, rng, prop_sd, acc):
        mu = self._mu(s)
        thr = s["thr"]
        lo, hi = self._bounds(thr)
        cur = np.sum(_log_prob_interval(lo - mu, hi - mu))
        prop = thr + prop_sd["thr"] * rng.standard_normal(thr.shape)
        ok = False
        if np.all(np.diff(prop) > 0):
            lo2, hi2 = self._bounds(prop)
            new = np.sum(_log_prob_interval(lo2 - mu, hi2 - mu))
            ok = np.log(rng.random()) < new - cur
            if ok:
                s["thr"], lo, hi = prop, lo2, hi2
        acc["thr"] = ok
        return rtruncnorm(mu, lo, hi, rng)

    def _scale_keys(self):
        return super()._scale_keys() + ("thr",)

    def _shift_keys(self):
        return super()._shift_keys() + ("thr",)

    def _extra_store(self, s):
        phi = self._phi(s)
        return {"thr": s["thr"] - phi.mean()}


class GaussianBYM2(SpatialRankLikelihood):
    """Gaussian BYM2 model on the raw outcome with an intercept and error SD.

    Written as z = (y - alpha) / sigma, which has unit error variance, so all
    the BYM2 and cluster steps are shared. alpha has a flat prior and log sigma
    a flat prior. Run with group_moves=False: z is determined by the data.
    """

    def _init_prop_sd(self):
        d = super()._init_prop_sd()
        d["sig"] = 0.02
        return d

    def _init_state(self, rng, disperse):
        s = super()._init_state(rng, disperse)
        s["alpha"] = self.y.mean()
        s["sigma"] = self.y.std() * (rng.uniform(0.7, 1.4) if disperse else 1.0)
        s["z"] = (self.y - s["alpha"]) / s["sigma"]
        return s

    def _update_latent(self, s, rng, prop_sd, acc):
        mu = self._mu(s)
        y = self.y
        s["alpha"] = rng.normal(np.mean(y - s["sigma"] * mu), s["sigma"] / np.sqrt(self.N))
        r = y - s["alpha"]

        def lp(sig):
            return -self.N * np.log(sig) - 0.5 * np.sum((r / sig - mu) ** 2)

        prop = s["sigma"] * np.exp(prop_sd["sig"] * rng.standard_normal())
        ok = np.log(rng.random()) < lp(prop) - lp(s["sigma"])
        if ok:
            s["sigma"] = prop
        acc["sig"] = ok
        return (y - s["alpha"]) / s["sigma"]

    def fit(self, *args, **kwargs):
        kwargs["group_moves"] = False
        return super().fit(*args, **kwargs)

    def _extra_store(self, s):
        phi = self._phi(s)
        return {"loc": s["alpha"] + s["sigma"] * phi.mean(), "sigma": s["sigma"]}


def prevalence_gaussian(model, cutoffs):
    """Area and overall prevalence of y < c under the Gaussian BYM2 model."""
    c = np.broadcast_to(np.asarray(cutoffs, float), (model.N,))
    mu = model.latent_means()
    loc = model.draws["loc"].reshape(-1, 1)
    sig = model.draws["sigma"].reshape(-1, 1)
    p = special.ndtr((c[None, :] - loc) / sig - mu)
    n = np.bincount(model.area, minlength=model.m)
    prev = np.stack([np.bincount(model.area, weights=row, minlength=model.m) for row in p]) / n
    return prev, p.mean(axis=1)


def prevalence_binary(model):
    """Area and overall prevalence of the lower category under the binary probit model."""
    mu = model.latent_means()
    thr = model.draws["thr"].reshape(-1, model.L - 1)[:, :1]
    p = special.ndtr(thr - mu)
    n = np.bincount(model.area, minlength=model.m)
    prev = np.stack([np.bincount(model.area, weights=row, minlength=model.m) for row in p]) / n
    return prev, p.mean(axis=1)


def _split(x):
    x = np.asarray(x, dtype=float)
    n = x.shape[1] // 2
    return np.vstack([x[:, :n], x[:, n:2 * n]])


def _rank_normalise(x):
    r = stats.rankdata(x, method="average").reshape(x.shape)
    return special.ndtri((r - 0.375) / (x.size + 0.25))


def _rhat_basic(x):
    C, n = x.shape
    W = x.var(axis=1, ddof=1).mean()
    B = n * x.mean(axis=1).var(ddof=1)
    var_hat = (n - 1) / n * W + B / n
    return np.sqrt(var_hat / W) if W > 0 else np.nan


def rhat(x):
    """Rank-normalised split R-hat (maximum of bulk and folded versions)."""
    xs = _split(x)
    bulk = _rhat_basic(_rank_normalise(xs))
    folded = _rhat_basic(_rank_normalise(np.abs(xs - np.median(xs))))
    return max(bulk, folded)


def _autocov(x):
    n = len(x)
    xc = x - x.mean()
    f = np.fft.rfft(xc, n=2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n] / n
    return ac


def _ess_basic(x):
    C, n = x.shape
    if n < 4:
        return np.nan
    acov = np.array([_autocov(c) for c in x])
    mean_var = acov[:, 0].mean() * n / (n - 1)
    var_plus = mean_var * (n - 1) / n
    if C > 1:
        var_plus += x.mean(axis=1).var(ddof=1)
    if var_plus <= 0:
        return np.nan
    rho = np.zeros(n)
    rho[0] = 1.0
    rho_even, rho_odd = 1.0, 1.0 - (mean_var - acov[:, 1].mean()) / var_plus
    rho[1] = rho_odd
    t = 1
    while t < n - 3 and rho_even + rho_odd > 0:
        rho_even = 1.0 - (mean_var - acov[:, t + 1].mean()) / var_plus
        rho_odd = 1.0 - (mean_var - acov[:, t + 2].mean()) / var_plus
        if rho_even + rho_odd >= 0:
            rho[t + 1], rho[t + 2] = rho_even, rho_odd
        t += 2
    max_t = t - 2 if t > 1 else 1
    for k in range(1, max_t - 1, 2):
        if rho[k + 1] + rho[k + 2] > rho[k - 1] + rho[k]:
            rho[k + 1] = rho[k + 2] = 0.5 * (rho[k - 1] + rho[k])
    tau = -1.0 + 2.0 * rho[: max_t + 1].sum() + (rho[max_t + 1] if max_t + 1 < n else 0.0)
    tau = max(tau, 1.0 / np.log10(C * n))
    return C * n / tau


def ess_bulk(x):
    return _ess_basic(_rank_normalise(_split(x)))


def ess_tail(x):
    xs = _split(x)
    q05, q95 = np.quantile(xs, [0.05, 0.95])
    return min(_ess_basic((xs <= q05).astype(float)), _ess_basic((xs <= q95).astype(float)))


def summarise(draws, names=None):
    """Table of mean, sd, 95% interval, R-hat and ESS for an array (chain, draw, k) or (chain, draw)."""
    d = np.asarray(draws)
    if d.ndim == 2:
        d = d[:, :, None]
    k = d.shape[2]
    names = names or [f"[{j}]" for j in range(k)]
    rows = []
    for j in range(k):
        x = d[:, :, j]
        flat = x.ravel()
        rows.append(dict(name=names[j], mean=flat.mean(), sd=flat.std(ddof=1),
                         q2_5=np.quantile(flat, 0.025), q97_5=np.quantile(flat, 0.975),
                         rhat=rhat(x), ess_bulk=ess_bulk(x), ess_tail=ess_tail(x)))
    return rows


def standardised_error(dist, size, rng):
    """Latent errors with mean 0 and variance 1."""
    if dist == "normal":
        return rng.standard_normal(size)
    if dist == "logistic":
        return rng.logistic(size=size) / (np.pi / np.sqrt(3))
    if dist == "t3":
        return rng.standard_t(3, size=size) / np.sqrt(3.0)
    if dist == "skewed":
        a = 5.0
        d = a / np.sqrt(1 + a * a)
        x = stats.skewnorm.rvs(a, size=size, random_state=rng)
        mean, sd = d * np.sqrt(2 / np.pi), np.sqrt(1 - 2 * d * d / np.pi)
        return (x - mean) / sd
    raise ValueError(dist)


def error_cdf(dist, x):
    """Distribution function of the standardised latent error."""
    x = np.asarray(x, dtype=float)
    if dist == "normal":
        return stats.norm.cdf(x)
    if dist == "logistic":
        return stats.logistic.cdf(x * np.pi / np.sqrt(3))
    if dist == "t3":
        return stats.t.cdf(x * np.sqrt(3.0), 3)
    if dist == "skewed":
        a = 5.0
        d = a / np.sqrt(1 + a * a)
        mean, sd = d * np.sqrt(2 / np.pi), np.sqrt(1 - 2 * d * d / np.pi)
        return stats.skewnorm.cdf(x * sd + mean, a)
    raise ValueError(dist)


def true_prevalence(d, cutoff):
    """True prevalence of y < cutoff for each sampled individual (continuous outcome).

    y = round(exp(0.25 z + 2.5), 1), so y < c  <=>  z < (log(c - 0.05) - 2.5) / 0.25
    for cut-offs on the 0.1 recording grid.
    """
    t = (np.log(np.asarray(cutoff, float) - 0.05) - 2.5) / 0.25
    return error_cdf(d["error"], (t - d["mu"]) / d["sd"])


def simulate(W, n_clusters_per_area=3, n_per_cluster=15, beta=(0.8, -0.5, 0.5, -0.3),
             sigma_phi=0.7, rho=0.8, sigma_eta=0.3, error="normal", outcome="continuous",
             n_categories=4, seed=0, log_sd_area=None):
    """Return a dict with y, X, area, cluster and the true latent quantities.

    Covariates: first half continuous N(0, 1), second half binary Bernoulli(0.5).
    log_sd_area: optional log standard deviation of the latent errors for each area
    (heteroscedastic latent errors); None gives unit standard deviation everywhere.
    """
    rng = np.random.default_rng(seed)
    g = AreaGraph(W)
    m = g.m
    beta = np.asarray(beta, dtype=float)
    p = len(beta)

    u = g.sample_icar(rng)
    v = rng.standard_normal(m)
    phi = sigma_phi * (np.sqrt(rho) * u + np.sqrt(1 - rho) * v)

    K = m * n_clusters_per_area
    cl_area = np.repeat(np.arange(m), n_clusters_per_area)
    cluster = np.repeat(np.arange(K), n_per_cluster)
    area = cl_area[cluster]
    N = len(cluster)
    eta = sigma_eta * rng.standard_normal(K)

    pc = (p + 1) // 2
    X = np.empty((N, p))
    X[:, :pc] = rng.standard_normal((N, pc))
    X[:, pc:] = rng.binomial(1, 0.5, (N, p - pc))

    mu = X @ beta + phi[area] + eta[cluster]
    sd = np.ones(N) if log_sd_area is None else np.exp(np.asarray(log_sd_area, float)[area])
    z = mu + sd * standardised_error(error, N, rng)

    if outcome == "continuous":
        y = np.round(np.exp(0.25 * z + 2.5), 1)
    elif outcome == "ordinal":
        cuts = np.quantile(z, np.linspace(0, 1, n_categories + 1)[1:-1])
        y = np.searchsorted(cuts, z).astype(float)
    else:
        raise ValueError(outcome)

    return dict(y=y, X=X, area=area, cluster=cluster, W=W, beta=beta, phi=phi,
                eta=eta, mu=mu, z=z, sd=sd, sigma_phi=sigma_phi, rho=rho, sigma_eta=sigma_eta,
                error=error)


SIM_CLUSTERS = {"small": 2, "large": 6}
SIM_ERRORS = ["normal", "logistic", "skewed", "t3"]
CUT_A = 11.0
CUT_B = (11.0, 10.5)
CUT_U = 20.0
CUT_U_TRANSFORM = 28.0
SIM_MCMC = dict(n_iter=2000, burn=700, n_chains=2)
LOG_SD_REGION = {"Tigray": 0.09, "Afar": 0.21, "Amhara": 0.04, "Oromia": 0.00, "Somali": 0.28,
                 "Benishangul-Gumuz": -0.03, "Central Ethiopia": 0.02, "Sidama": -0.05,
                 "South West Ethiopia": -0.09, "South Ethiopia": -0.04, "Gambela": 0.09,
                 "Harari": 0.19, "Addis Ababa": 0.03, "Dire Dawa": 0.18}
SPREAD_REGIONS = [r for r in LOG_SD_REGION if r != "Oromia"]

_SIM_DATA = {}


def sim_data(data_dir="data"):
    """Adjacency matrices of the 14 regions and the 107 zones, and the region of each zone."""
    if not _SIM_DATA:
        _SIM_DATA["graphs"] = {"regions": np.load(os.path.join(data_dir, "W_regions.npy")),
                               "zones": np.load(os.path.join(data_dir, "W_zones.npy"))}
        _SIM_DATA["zone_region"] = (pd.read_csv(os.path.join(data_dir, "zones_lookup.csv"))["adm1_name"]
                                    .replace({"Contested": "Tigray"}).to_numpy())
    return _SIM_DATA


def scenarios():
    out = []
    for g in ["regions", "zones"]:
        for size in SIM_CLUSTERS:
            for e in SIM_ERRORS:
                out.append(("continuous", g, size, e))
    for e in ["normal", "skewed"]:
        out.append(("ordinal", "zones", "large", e))
    for size in SIM_CLUSTERS:
        for pattern in ["equal", "unequal"]:
            out.append(("spread", "zones", size, pattern))
    out.append(("transform", "zones", "small", "normal"))
    return out


def scenario_seed(sc, rep):
    return (zlib.crc32("|".join(sc).encode()) % 10_000) * 1_000 + rep


def _summ(draws_area, truth, label):
    lo, hi = np.quantile(draws_area, [0.025, 0.975], axis=0)
    est = draws_area.mean(axis=0)
    return {f"{label}_err": est - truth,
            f"{label}_cov": ((lo <= truth) & (truth <= hi)).astype(float),
            f"{label}_width": hi - lo}


def _common(out, d, true_ratio):
    ratio = (out["beta"][:, :, 1] / out["beta"][:, :, 0]).ravel()
    lo, hi = np.quantile(ratio, [0.025, 0.975])
    rho = out["rho"].ravel()
    rlo, rhi = np.quantile(rho, [0.025, 0.975])
    return dict(ratio_err=ratio.mean() - true_ratio, ratio_cov=float(lo <= true_ratio <= hi),
                ratio_width=hi - lo,
                rank_corr=stats.spearmanr(out["phi"].mean(axis=(0, 1)), d["phi"]).statistic,
                rho_err=rho.mean() - d["rho"], rho_cov=float(rlo <= d["rho"] <= rhi))


def _add_prevalences(r, pA, pB, pU, tA, tB, tU):
    r.update(_summ(pA, tA, "A")); r.update(_summ(pB, tB, "B")); r.update(_summ(pB - pA, tB - tA, "D"))
    if pU is not None:
        r.update(_summ(pU, tU, "U"))
    return r


def boxcox(y, lam):
    return np.log(y) if abs(lam) < 1e-8 else (y ** lam - 1.0) / lam


def boxcox_lambda(y, X, area, m):
    """Profile-likelihood estimate of lambda for y_lambda ~ X + area indicators."""
    D = np.column_stack([X, np.eye(m)[area]])
    slog = np.log(y).sum()
    N = len(y)

    def negll(lam):
        z = boxcox(y, lam)
        coef, *_ = np.linalg.lstsq(D, z, rcond=None)
        rss = np.sum((z - D @ coef) ** 2)
        return 0.5 * N * np.log(rss / N) - (lam - 1.0) * slog

    grid = np.linspace(-2, 2, 41)
    l0 = grid[np.argmin([negll(l) for l in grid])]
    res = optimize.minimize_scalar(negll, bounds=(max(-2.0, l0 - 0.1), min(2.0, l0 + 0.1)), method="bounded")
    return float(res.x)


def boxcox_offset(y):
    """Box-Cox needs y > 0: if the outcome is recorded as zero (rounding), add half the recording unit."""
    return 0.05 if np.min(y) <= 0 else 0.0


def prevalence_boxcox(model, cutoffs, lam, off=0.0):
    """Area prevalence of y < c under the Box-Cox Gaussian model (continuity correction 0.05)."""
    c = np.broadcast_to(np.asarray(cutoffs, float) - 0.05 + off, (model.N,))
    hc = boxcox(c, lam)
    mu = model.latent_means()
    loc = model.draws["loc"].reshape(-1, 1)
    sig = model.draws["sigma"].reshape(-1, 1)
    p = special.ndtr((hc[None, :] - loc) / sig - mu)
    n = np.bincount(model.area, minlength=model.m)
    return np.stack([np.bincount(model.area, weights=row, minlength=model.m) for row in p]) / n


def _fit_all(d, W, seed, cA, cB, cU, tA, tB, tU, Z=None, with_binary=True, log_sd_true=None):
    """Fit every model to one simulated dataset and summarise prevalence errors."""
    true_ratio = d["beta"][1] / d["beta"][0]
    args = (d["X"], d["area"], d["cluster"], W)
    m = W.shape[0]
    res = {}

    def srl(Zs):
        t0 = time.time()
        mod = SpatialRankLikelihood(d["y"], *args, Z_scale=Zs)
        out = mod.fit(**SIM_MCMC, seed=seed)
        pA, _ = prevalence(mod, cA, n_draws=200, seed=seed)
        pB, _ = prevalence(mod, cB, n_draws=200, seed=seed)
        pU, _ = prevalence(mod, cU, n_draws=200, seed=seed, below=False)
        r = _add_prevalences(_common(out, d, true_ratio), pA, pB, pU, tA, tB, tU)
        if Zs is not None:
            g = out["gamma"].reshape(-1, Zs.shape[1])
            lo, hi = np.quantile(g, [0.025, 0.975], axis=0)
            gt = np.array([log_sd_true[r_] for r_ in SPREAD_REGIONS])
            r["gamma_err"] = g.mean(0) - gt
            r["gamma_cov"] = ((lo <= gt) & (gt <= hi)).astype(float)
        r["seconds"] = time.time() - t0
        return r

    if Z is not None:
        res["SRL-LS"] = srl(Z)
    res["SRL"] = srl(None)

    t0 = time.time()
    mod = GaussianBYM2(d["y"], *args)
    out = mod.fit(**SIM_MCMC, seed=seed)
    pA, _ = prevalence_gaussian(mod, cA)
    pB, _ = prevalence_gaussian(mod, cB)
    pU = 1 - prevalence_gaussian(mod, cU)[0]
    res["Gaussian"] = _add_prevalences(_common(out, d, true_ratio), pA, pB, pU, tA, tB, tU)
    res["Gaussian"]["seconds"] = time.time() - t0

    t0 = time.time()
    try:
        off = boxcox_offset(d["y"])
        lam = boxcox_lambda(d["y"] + off, d["X"], d["area"], m)
        mod = GaussianBYM2(boxcox(d["y"] + off, lam), *args)
        out = mod.fit(**SIM_MCMC, seed=seed)
        pA = prevalence_boxcox(mod, cA, lam, off)
        pB = prevalence_boxcox(mod, cB, lam, off)
        pU = 1 - prevalence_boxcox(mod, cU, lam, off)
        r = _add_prevalences(_common(out, d, true_ratio), pA, pB, pU, tA, tB, tU)
        r["lambda"], r["offset"], r["seconds"] = lam, off, time.time() - t0
        res["BoxCox"] = r
    except Exception as exc:
        res["BoxCox"] = {"failed": repr(exc)}

    if with_binary:
        t0 = time.time()
        fits = {}
        for lab, c in [("A", cA), ("B", cB)]:
            mod = OrdinalProbitBYM2((d["y"] >= c).astype(float), *args)
            out = mod.fit(**SIM_MCMC, seed=seed)
            fits[lab] = (out, prevalence_binary(mod)[0])
        pA, pB = fits["A"][1], fits["B"][1]
        k = min(len(pA), len(pB))
        r = _common(fits["A"][0], d, true_ratio)
        r.update(_summ(pA, tA, "A")); r.update(_summ(pB, tB, "B")); r.update(_summ(pB[:k] - pA[:k], tB - tA, "D"))
        r["seconds"] = time.time() - t0
        res["Binary"] = r
    return res


def _truths(d, m, true_cdf, cA, cB, cU):
    area = d["area"]
    n_area = np.bincount(area, minlength=m)
    tA = np.bincount(area, weights=true_cdf(cA), minlength=m) / n_area
    tB = np.bincount(area, weights=true_cdf(cB), minlength=m) / n_area
    tU = np.bincount(area, weights=1 - true_cdf(cU), minlength=m) / n_area
    return tA, tB, tU


def run_continuous(graph, size, error, seed, pattern=None, data_dir="data"):
    S = sim_data(data_dir)
    W = S["graphs"][graph]
    m = W.shape[0]
    log_sd_area, Z, log_sd_true = None, None, None
    if pattern is not None:
        log_sd_true = LOG_SD_REGION if pattern == "unequal" else {r: 0.0 for r in LOG_SD_REGION}
        log_sd_area = np.array([log_sd_true[r] for r in S["zone_region"]])
        error = "normal"
    d = simulate(W, n_clusters_per_area=SIM_CLUSTERS[size], error=error, outcome="continuous", seed=seed,
                 log_sd_area=log_sd_area)
    if pattern is not None:
        reg_ind = S["zone_region"][d["area"]]
        Z = np.stack([(reg_ind == r).astype(float) for r in SPREAD_REGIONS], axis=1)
    cB = np.where(d["X"][:, 3] == 1, CUT_B[1], CUT_B[0])
    tA, tB, tU = _truths(d, m, lambda c: true_prevalence(d, c), CUT_A, cB, CUT_U)
    return _fit_all(d, W, seed, CUT_A, cB, CUT_U, tA, tB, tU, Z=Z, with_binary=pattern is None,
                    log_sd_true=log_sd_true)


def g_transform(z):
    """Transformation outside the Box-Cox family: log y is piecewise linear in z with a kink."""
    return np.round(np.where(z < 0, np.exp(2.5 + 0.25 * z), np.exp(2.5 + 0.45 * z)), 1)


def g_transform_inverse(c):
    lc = np.log(np.asarray(c, float) - 0.05) - 2.5
    return np.where(lc < 0, lc / 0.25, lc / 0.45)


def run_transform(seed, data_dir="data"):
    W = sim_data(data_dir)["graphs"]["zones"]
    m = W.shape[0]
    d = simulate(W, n_clusters_per_area=SIM_CLUSTERS["small"], error="normal", outcome="continuous", seed=seed)
    d["y"] = g_transform(d["z"])
    cB = np.where(d["X"][:, 3] == 1, CUT_B[1], CUT_B[0])
    true_cdf = lambda c: error_cdf("normal", (g_transform_inverse(c) - d["mu"]) / d["sd"])
    tA, tB, tU = _truths(d, m, true_cdf, CUT_A, cB, CUT_U_TRANSFORM)
    return _fit_all(d, W, seed, CUT_A, cB, CUT_U_TRANSFORM, tA, tB, tU)


def run_ordinal(graph, size, error, seed, data_dir="data"):
    W = sim_data(data_dir)["graphs"][graph]
    d = simulate(W, n_clusters_per_area=SIM_CLUSTERS[size], error=error, outcome="ordinal", seed=seed)
    true_ratio = d["beta"][1] / d["beta"][0]
    res = {}
    for name, M in [("SRL", SpatialRankLikelihood), ("OrdinalProbit", OrdinalProbitBYM2)]:
        t0 = time.time()
        mod = M(d["y"], d["X"], d["area"], d["cluster"], W)
        out = mod.fit(**SIM_MCMC, seed=seed)
        r = _common(out, d, true_ratio)
        r["beta_mean"] = out["beta"].mean(axis=(0, 1))
        r["seconds"] = time.time() - t0
        res[name] = r
    return res


def run_one(sc, rep, outdir, data_dir="data"):
    path = os.path.join(outdir, "__".join(sc) + f"__{rep:04d}.pkl")
    if os.path.exists(path):
        return "skipped"
    seed = scenario_seed(sc, rep)
    block, graph, size, error = sc
    try:
        if block == "continuous":
            res = run_continuous(graph, size, error, seed, data_dir=data_dir)
        elif block == "spread":
            res = run_continuous(graph, size, "normal", seed, pattern=error, data_dir=data_dir)
        elif block == "transform":
            res = run_transform(seed, data_dir=data_dir)
        else:
            res = run_ordinal(graph, size, error, seed, data_dir=data_dir)
    except Exception as exc:
        res = {"failed": repr(exc)}
    with open(path + ".tmp", "wb") as f:
        pickle.dump(dict(scenario=sc, rep=rep, seed=seed, results=res), f)
    os.replace(path + ".tmp", path)
    return "done"


def run_simulation(reps=100, jobs=-1, outdir="sim_results", data_dir="data", only=None):
    """Run (or resume) the simulation study; one result file per scenario and replicate."""
    from joblib import Parallel, delayed
    os.makedirs(outdir, exist_ok=True)
    jobs_list = [(sc, r) for r in range(reps) for sc in scenarios()
                 if only is None or only in "__".join(sc)]
    t0 = time.time()
    status = Parallel(n_jobs=jobs, verbose=10)(delayed(run_one)(sc, r, outdir, data_dir) for sc, r in jobs_list)
    print(f"{status.count('done')} run, {status.count('skipped')} skipped, {time.time() - t0:.0f} s")


def summarise_simulation(folder="sim_results", out_csv="simulation_summary.csv"):
    """One row per scenario x model. Coverage Monte Carlo SEs are computed across replicates,
    since areas within a replicate share one fitted model."""
    recs = []
    for f in sorted(glob.glob(os.path.join(folder, "*.pkl"))):
        with open(f, "rb") as fh:
            recs.append(pickle.load(fh))
    rows, failed, by_sc = [], 0, {}
    for r in recs:
        if "failed" in r["results"]:
            failed += 1
            continue
        by_sc.setdefault(tuple(r["scenario"]), []).append(r["results"])
    for sc, reps in sorted(by_sc.items()):
        block, graph, size, error = sc
        models = list(dict.fromkeys(k for x in reps for k in x))
        for model in models:
            rs = [x[model] for x in reps if model in x and "failed" not in x[model]]
            if not rs:
                continue
            row = dict(block=block, graph=graph, size=size, error=error, model=model, reps=len(rs))
            re_ = np.array([x["ratio_err"] for x in rs])
            row.update(ratio_bias=re_.mean(), ratio_rmse=np.sqrt((re_ ** 2).mean()),
                       ratio_cov=np.mean([x["ratio_cov"] for x in rs]),
                       rank_corr=np.mean([x["rank_corr"] for x in rs]),
                       rho_cov=np.mean([x["rho_cov"] for x in rs]),
                       seconds=np.mean([x["seconds"] for x in rs]))
            for lab in ["A", "B", "D", "U"]:
                if f"{lab}_err" not in rs[0]:
                    continue
                e = np.concatenate([x[f"{lab}_err"] for x in rs])
                cov_rep = np.array([x[f"{lab}_cov"].mean() for x in rs])
                row[f"{lab}_bias"] = e.mean()
                row[f"{lab}_rmse"] = np.sqrt((e ** 2).mean())
                row[f"{lab}_cov"] = cov_rep.mean()
                row[f"{lab}_cov_mcse"] = cov_rep.std(ddof=1) / np.sqrt(len(rs)) if len(rs) > 1 else np.nan
                row[f"{lab}_width"] = np.mean([x[f"{lab}_width"].mean() for x in rs])
            if "gamma_err" in rs[0]:
                ge = np.concatenate([x["gamma_err"] for x in rs])
                row["gamma_rmse"] = np.sqrt((ge ** 2).mean())
                row["gamma_cov"] = np.mean([x["gamma_cov"].mean() for x in rs])
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"{len(df)} rows written to {out_csv}; failed replicates: {failed}")
    return df



def simulation_key_numbers(summary_csv="simulation_summary.csv", out_csv="simulation_key_numbers.csv"):
    """Ratios of root mean squared errors to the rank-likelihood model, coverage ranges by error
    distribution, and the bias and interval width behind the coverage of the difference."""
    df = pd.read_csv(summary_csv)
    cont = df[df.block == "continuous"]
    k = cont.pivot_table(index=["graph", "size", "error"], columns="model",
                         values=["A_rmse", "D_rmse", "U_rmse"])
    rows = []
    for q in ["A_rmse", "D_rmse", "U_rmse"]:
        for m in ["BoxCox", "Gaussian", "Binary"]:
            if m not in k[q] or k[q][m].isna().all():
                continue
            r = k[q][m] / k[q]["SRL"]
            rows.append(dict(quantity=q, model=m, ratio_min=r.min(), ratio_median=r.median(), ratio_max=r.max()))
    ratios = pd.DataFrame(rows)
    cov = (cont.groupby(["model", "error"])[["A_cov", "D_cov", "U_cov"]].agg(["min", "max"]).round(2))
    srl = cont[cont.model == "SRL"][["graph", "size", "error", "D_bias", "D_rmse", "D_width", "D_cov", "D_cov_mcse"]].copy()
    for c in ["D_bias", "D_rmse", "D_width"]:
        srl[c] = 100 * srl[c]
    mcse = df[df.block.isin(["continuous", "spread", "transform"])].groupby("model")[
        ["A_cov_mcse", "D_cov_mcse", "U_cov_mcse"]].max()
    ratios.to_csv(out_csv, index=False)
    pd.set_option("display.width", 200)
    print("RMSE ratio to the rank-likelihood model (16 continuous scenarios):")
    print(ratios.round(2).to_string(index=False))
    print("\nCoverage range by model and error distribution:")
    print(cov.to_string())
    print("\nRank likelihood, difference B - A (percentage points):")
    print(srl.round(3).to_string(index=False))
    print("\nLargest Monte Carlo standard error of coverage, by model:")
    print(mcse.round(3).to_string())
    return ratios


def simulation_outputs(summary_csv="simulation_summary.csv", fig_dir="figures", tab_dir="tables"):
    """Figure 1 and the LaTeX rows of the simulation tables."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    os.makedirs(fig_dir, exist_ok=True); os.makedirs(tab_dir, exist_ok=True)
    df = pd.read_csv(summary_csv)
    ERR_LAB = {"normal": "Normal", "logistic": "Logistic", "skewed": "Skew-normal", "t3": "$t_3$"}
    ERR_TICK = {"normal": "N", "logistic": "L", "skewed": "S", "t3": "t"}
    GROUPS = [("regions", "small", "Regions, 2"), ("regions", "large", "Regions, 6"),
              ("zones", "small", "Zones, 2"), ("zones", "large", "Zones, 6")]
    MODELS = ["SRL", "BoxCox", "Gaussian", "Binary"]
    MOD_LAB = {"SRL": "Rank likelihood", "BoxCox": "Box--Cox Gaussian", "Gaussian": "Gaussian",
               "Binary": "Binary probit"}
    COL = {"SRL": "#d62728", "Gaussian": "#1f77b4", "Binary": "#2ca02c", "BoxCox": "#9467bd"}
    MARK = {"SRL": "o", "Gaussian": "s", "Binary": "^", "BoxCox": "D"}
    cont = df[df.block == "continuous"]

    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(3, 2, figsize=(7.5, 7.2), sharex=True)
    quant = [("A", "Definition A"), ("D", "Difference B $-$ A"), ("U", "Upper tail")]
    xpos, x = {}, 0
    for g, s, _ in GROUPS:
        for e in SIM_ERRORS:
            xpos[(g, s, e)] = x
            x += 1
        x += 0.8
    for r, (q, qlab) in enumerate(quant):
        for c, (stat, slab) in enumerate([("rmse", "RMSE (pp)"), ("cov", "Coverage")]):
            a = ax[r, c]
            for k, mname in enumerate(MODELS):
                sub = cont[cont.model == mname]
                col = f"{q}_{stat}"
                if col not in sub or sub[col].isna().all():
                    continue
                xs = [xpos[(t.graph, t.size, t.error)] + (k - 1.5) * 0.19 for t in sub.itertuples()]
                vals = sub[col] * (100 if stat == "rmse" else 1)
                a.scatter(xs, vals, s=30, marker=MARK[mname], color=COL[mname], edgecolor="white",
                          linewidth=0.5, label=MOD_LAB[mname].replace("--", "\u2013"), zorder=3)
            if stat == "cov":
                a.axhline(0.95, color="0.4", linewidth=0.8, linestyle="--", zorder=1)
                a.set_ylim(-0.03, 1.03)
            else:
                a.set_yscale("log")
                a.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 2, 3, 5)))
                a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
                a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            a.set_ylabel(f"{qlab}\n{slab}" if c == 0 else slab)
            a.grid(axis="y", color="0.9", linewidth=0.5)
            for sp in ["top", "right"]:
                a.spines[sp].set_visible(False)
    for c in range(2):
        ax[2, c].set_xticks(list(xpos.values()))
        ax[2, c].set_xticklabels([ERR_TICK[k[2]] for k in xpos])
        for g, s, lab in GROUPS:
            mid = np.mean([xpos[(g, s, e)] for e in SIM_ERRORS])
            ax[2, c].annotate(lab, (mid, 0), xycoords=("data", "axes fraction"), xytext=(0, -24),
                              textcoords="offset points", ha="center", fontsize=8)
    fig.align_ylabels(ax[:, 0])
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(os.path.join(fig_dir, "fig_simulation.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    f2 = lambda v: "--" if pd.isna(v) else f"{v:.2f}"
    f3 = lambda v: "--" if pd.isna(v) else f"{100 * v:.2f}"
    with open(os.path.join(tab_dir, "sim_continuous_rows.tex"), "w") as f:
        for g, s, glab in GROUPS:
            for e in SIM_ERRORS:
                for mname in MODELS:
                    t = cont[(cont.graph == g) & (cont["size"] == s) & (cont.error == e) & (cont.model == mname)]
                    if t.empty:
                        continue
                    t = t.iloc[0]
                    lead = f"{glab} & {ERR_LAB[e]}" if mname == "SRL" else " & "
                    f.write(f"{lead} & {MOD_LAB[mname]} & {f3(t.A_rmse)} & {f2(t.A_cov)} & {f3(t.D_rmse)} & "
                            f"{f2(t.D_cov)} & {f3(t.U_rmse)} & {f2(t.U_cov)} & {f2(t.rank_corr)} \\\\\n")
                f.write("\\addlinespace\n")
    rng_ = lambda p, fmt: "--" if pd.isna(p[0]) else (fmt(p[0]) if fmt(p[0]) == fmt(p[1]) else f"{fmt(p[0])}--{fmt(p[1])}")
    with open(os.path.join(tab_dir, "sim_summary_rows.tex"), "w") as f:
        for mname in MODELS:
            for gauss in [True, False]:
                t = cont[(cont.model == mname) & ((cont.error == "normal") == gauss)]
                cells = []
                for q in ["A", "D", "U"]:
                    for st in ["rmse", "cov"]:
                        v = t[f"{q}_{st}"]
                        cells.append(rng_((v.min(), v.max()) if v.notna().any() else (np.nan, np.nan),
                                          f3 if st == "rmse" else f2))
                f.write(f"{MOD_LAB[mname]} & {'Normal' if gauss else 'Non-normal'} & " + " & ".join(cells) + " \\\\\n")
    sp = df[df.block == "spread"]
    with open(os.path.join(tab_dir, "sim_spread_rows.tex"), "w") as f:
        for spread, slab in [("equal", "Equal"), ("unequal", "Varying by region")]:
            for s, nlab in [("small", "2"), ("large", "6")]:
                for k, mname in enumerate(["SRL-LS", "SRL", "BoxCox", "Gaussian"]):
                    t = sp[(sp.error == spread) & (sp["size"] == s) & (sp.model == mname)]
                    if t.empty:
                        continue
                    t = t.iloc[0]
                    lab = {"SRL-LS": "Location--scale", "SRL": "Location-only", "BoxCox": "Box--Cox Gaussian",
                           "Gaussian": "Gaussian"}[mname]
                    lead = f"{slab} & {nlab}" if k == 0 else " & "
                    f.write(f"{lead} & {lab} & {f3(t.A_rmse)} & {f2(t.A_cov)} & {f3(t.D_rmse)} & {f2(t.D_cov)} & "
                            f"{f3(t.U_rmse)} & {f2(t.U_cov)} \\\\\n")
            f.write("\\addlinespace\n")
    tr = df[df.block == "transform"]
    with open(os.path.join(tab_dir, "sim_transform_rows.tex"), "w") as f:
        for mname in MODELS:
            t = tr[tr.model == mname]
            if t.empty:
                continue
            t = t.iloc[0]
            f.write(f"{MOD_LAB[mname]} & {f3(t.A_rmse)} & {f2(t.A_cov)} & {f3(t.D_rmse)} & {f2(t.D_cov)} & "
                    f"{f3(t.U_rmse)} & {f2(t.U_cov)} \\\\\n")
    print("figure and table rows written to", fig_dir, "and", tab_dir)


REGION_NAMES = {1: "Tigray", 2: "Afar", 3: "Amhara", 4: "Oromia", 5: "Somali", 6: "Benishangul-Gumuz",
                7: "Central Ethiopia", 8: "Sidama", 9: "South West Ethiopia", 10: "South Ethiopia",
                12: "Gambella", 13: "Harari", 14: "Addis Ababa", 15: "Dire Dawa"}


def _paths(root):
    p = dict(root=root, app=os.path.join(root, "data", "application"), fig=os.path.join(root, "figures"),
             tab=os.path.join(root, "tables"), admin=os.path.join(root, "data", "admin", "eth_admin_boundaries.gdb"))
    for k in ["app", "fig", "tab"]:
        os.makedirs(p[k], exist_ok=True)
    return p


def extract_data(root=".", ir=None, gps=None, lms=None, adm=None):
    """Build the labelled analysis file (Stata 14) from the DHS recode and GPS files.

    Outcome: BMI (kg/m2). Definition A uses adult cut-offs for every woman (thinness < 18.5,
    overweight >= 25, obesity >= 30); definition B uses WHO 2007 BMI-for-age for women younger
    than 19 years (thinness z < -2, overweight z > +1, obesity z > +2), converted to individual
    BMI cut-offs with the LMS method. Women who are pregnant or less than two months postpartum
    are flagged (in_sample = 0).
    """
    import pyreadstat
    import geopandas as gpd
    from libpysal.weights import Queen
    P = _paths(root)
    IR = ir or os.path.join(root, "data", "dhs", "ETIR8AFL.dta")
    GPS = gps or os.path.join(root, "data", "dhs", "ETGE8AFL.shp")
    LMS = lms or os.path.join(root, "data", "who2007", "bfawho2007.txt")
    ADM = adm or P["admin"]
    OUT = os.path.join(P["app"], "ethiopia_women_bmi_2024.dta")

    COLS = ["caseid", "v001", "v002", "v003", "v005", "v008", "v011", "v012", "v021", "v022", "v023",
            "v024", "v025", "v106", "v133", "v190", "v191", "v130", "v501", "v201", "v222", "v213", "v404",
            "v136", "v151", "v714", "v113", "v116", "v161", "v313", "v463z", "v157", "v158", "v159",
            "v437", "v438", "v445"]
    ir_, meta = pyreadstat.read_dta(IR, usecols=COLS, apply_value_formats=False)
    VL = meta.variable_value_labels
    for c in COLS[1:]:
        ir_[c] = pd.to_numeric(ir_[c], errors="coerce").astype(float)

    bmi = ir_["v445"].where(ir_["v445"] < 9990) / 100
    d = ir_[bmi.notna()].copy()
    d["bmi"] = (bmi[bmi.notna()]).round(2)
    d["weight_kg"] = (d["v437"].where(d["v437"] < 9990) / 10).round(1)
    d["height_cm"] = (d["v438"].where(d["v438"] < 9990) / 10).round(1)
    d["pregnant"] = (d["v213"] == 1).astype(int)
    d["postpartum_lt2m"] = (d["v222"] < 2).fillna(False).astype(int)
    d["in_sample"] = ((d["pregnant"] == 0) & (d["postpartum_lt2m"] == 0)).astype(int)
    d["excl_reason"] = np.select([d["pregnant"] == 1, d["postpartum_lt2m"] == 1], [1, 2], 0)

    d["age"] = d["v012"]
    d["age_months"] = d["v008"] - d["v011"]
    ref = pd.read_csv(LMS, sep=r"\s+")
    ref = ref[ref["sex"] == 2].set_index("age")[["l", "m", "s"]]

    def lms_at(age_m):
        """WHO AnthroPlus rule: linear interpolation of L, M, S between whole months."""
        lo = np.floor(age_m)
        frac = age_m - lo
        a = ref.reindex(lo).to_numpy()
        b = ref.reindex(lo + 1).to_numpy()
        b = np.where(np.isnan(b), a, b)
        return [a[:, k] + frac * (b[:, k] - a[:, k]) for k in range(3)]

    adol = (d["age_months"] >= 60) & (d["age_months"] < 228)
    L, M, S = lms_at(d["age_months"].where(adol).to_numpy(float))
    d["bmi_z"] = np.where(adol, ((d["bmi"] / M) ** L - 1) / (L * S), np.nan).round(2)
    bmi_at_z = lambda z: M * (1 + L * S * z) ** (1 / L)
    d["who_ref"] = adol.astype(int)
    d["cut_thin_A"], d["cut_over_A"], d["cut_obese_A"] = 18.5, 25.0, 30.0
    d["cut_thin_B"] = np.where(adol, bmi_at_z(-2.0), 18.5).round(2)
    d["cut_over_B"] = np.where(adol, bmi_at_z(1.0), 25.0).round(2)
    d["cut_obese_B"] = np.where(adol, bmi_at_z(2.0), 30.0).round(2)
    for k in ["thin", "over", "obese"]:
        for dfn in "AB":
            c = d[f"cut_{k}_{dfn}"]
            d[f"{k}_{dfn}"] = ((d["bmi"] < c) if k == "thin" else (d["bmi"] >= c)).astype(int)

    d["cluster"], d["household"], d["line"] = d["v001"], d["v002"], d["v003"]
    d["wt"] = (d["v005"] / 1e6).round(6)
    d["psu"], d["strata"], d["stratum_design"] = d["v021"], d["v022"], d["v023"]
    d["region"], d["residence"] = d["v024"], d["v025"]
    g = gpd.read_file(GPS)[["DHSCLUST", "ALT_DEM", "LATNUM", "LONGNUM"]]
    g = g.rename(columns={"DHSCLUST": "cluster", "ALT_DEM": "altitude_m", "LATNUM": "latitude", "LONGNUM": "longitude"})
    g["cluster"] = g["cluster"].astype(int)
    g["altitude_m"] = g["altitude_m"].where(g["altitude_m"] != 9999)
    zones = gpd.read_file(ADM, layer="eth_admin2").sort_values("adm2_pcode").reset_index(drop=True)
    zones["zone"] = np.arange(1, len(zones) + 1)
    pts = gpd.read_file(GPS)[["DHSCLUST", "DHSREGNA", "geometry"]].to_crs(zones.crs)
    pts["cluster"] = pts["DHSCLUST"].astype(int)
    j = gpd.sjoin(pts, zones[["zone", "adm1_name", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated()]
    norm = lambda x: x.fillna("").str.lower().str.replace(r"[^a-z]", "", regex=True).str.replace("gambela", "gambella")
    ok = j["zone"].notna() & ((norm(j["DHSREGNA"]) == norm(j["adm1_name"])) | (j["adm1_name"] == "Contested"))
    j["zone_assign"] = np.where(ok, 0, 1)
    zp, jp = zones.to_crs(32637), j.to_crs(32637)
    for i in j.index[~ok]:
        cand = zp[norm(zp["adm1_name"]) == norm(pd.Series([j.at[i, "DHSREGNA"]])).iloc[0]]
        j.at[i, "zone"] = cand.loc[cand.distance(jp.at[i, "geometry"]).idxmin(), "zone"]
    g = g.merge(j[["cluster", "zone", "zone_assign"]], on="cluster", how="left")
    d = d.merge(g, on="cluster", how="left", validate="many_to_one")
    d["zone"] = d["zone"].astype(int)
    w = Queen.from_dataframe(zones, use_index=False)
    Wz = np.zeros((len(zones), len(zones)))
    for i, nb in w.neighbors.items():
        Wz[i, nb] = 1
    np.save(os.path.join(P["app"], "W_zones.npy"), Wz)
    zones[["zone", "adm2_pcode", "adm2_name", "adm1_name"]].to_csv(os.path.join(P["app"], "zones_lookup.csv"), index=False)

    d["age_group"] = pd.cut(d["age"], [14, 19, 24, 29, 34, 39, 44, 49], labels=False) + 1
    d["education"] = d["v106"]
    d["educ_years"] = d["v133"].where(d["v133"] < 90)
    d["wealth"], d["wealth_score"] = d["v190"], (d["v191"] / 1e5).round(5)
    d["religion"] = d["v130"].replace({2: 5, 5: 5, 96: 5}).map({1: 1, 3: 2, 4: 3, 5: 4})
    d["marital"] = d["v501"].map({0: 1, 1: 2, 2: 2, 3: 3, 4: 3, 5: 3})
    d["parity"] = d["v201"]
    d["parity_group"] = pd.cut(d["v201"], [-1, 0, 2, 4, 99], labels=False) + 1
    d["breastfeeding"] = d["v404"]
    d["hh_size"] = d["v136"]
    d["female_head"] = (d["v151"] == 2).astype(int)
    d["working"] = d["v714"]
    d["media_weekly"] = ((d[["v157", "v158", "v159"]] == 2).any(axis=1)).astype(int)
    d["modern_contra"] = (d["v313"] == 3).astype(int)
    d["tobacco"] = (d["v463z"] == 0).astype(int)
    d["water"] = np.select([d["v113"].isin({11, 12, 13, 14, 21, 31, 41, 51, 61, 62, 71, 72}),
                            d["v113"].isin({32, 42, 43, 96})], [1, 2], np.nan)
    d["sanitation"] = np.select([d["v116"].isin({11, 12, 13, 15, 21, 22, 41}),
                                 d["v116"].isin({14, 23, 42, 43, 96}), d["v116"] == 31], [1, 2, 3], np.nan)
    d["clean_fuel"] = np.select([d["v161"].isin({1, 2, 3, 4, 5, 6}),
                                 d["v161"].isin({7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 96}), d["v161"] == 95],
                                [1, 0, 2], np.nan)

    KEEP = ["caseid", "cluster", "household", "line", "wt", "psu", "strata", "stratum_design", "region",
            "residence", "zone", "zone_assign", "latitude", "longitude", "altitude_m", "in_sample", "excl_reason",
            "pregnant", "postpartum_lt2m", "bmi", "weight_kg", "height_cm", "age", "age_months", "age_group",
            "who_ref", "bmi_z", "cut_thin_A", "cut_over_A", "cut_obese_A", "cut_thin_B", "cut_over_B",
            "cut_obese_B", "thin_A", "over_A", "obese_A", "thin_B", "over_B", "obese_B", "education",
            "educ_years", "wealth", "wealth_score", "religion", "marital", "parity", "parity_group",
            "breastfeeding", "hh_size", "female_head", "working", "media_weekly", "modern_contra", "tobacco",
            "water", "sanitation", "clean_fuel"]
    out = d[KEEP].sort_values(["cluster", "household", "line"]).reset_index(drop=True)
    for c in ["zone", "zone_assign", "cluster", "household", "line", "psu", "strata", "stratum_design", "region",
              "residence", "in_sample", "excl_reason", "pregnant", "postpartum_lt2m", "age", "age_months",
              "age_group", "who_ref", "thin_A", "over_A", "obese_A", "thin_B", "over_B", "obese_B", "education",
              "wealth", "religion", "marital", "parity", "parity_group", "breastfeeding", "hh_size", "female_head",
              "working", "media_weekly", "modern_contra", "tobacco"]:
        out[c] = out[c].astype("int16" if out[c].max() < 30000 else "int32")
    for c in ["water", "sanitation", "clean_fuel", "educ_years"]:
        out[c] = out[c].astype("float32")
    VAR_LABELS = {
        "caseid": "Case identification", "cluster": "Cluster number", "household": "Household number",
        "line": "Respondent's line number", "wt": "Women's sample weight", "psu": "Primary sampling unit",
        "strata": "Sample strata for sampling errors", "stratum_design": "Stratum used in sample design (region x residence)",
        "region": "Region", "residence": "Place of residence",
        "zone": "Zone (administrative level 2, OCHA boundaries 2025)",
        "zone_assign": "Zone assignment method",
        "latitude": "Cluster latitude (displaced, WGS84)", "longitude": "Cluster longitude (displaced, WGS84)",
        "altitude_m": "Cluster altitude (m, digital elevation model)",
        "in_sample": "Analysis sample (not pregnant, not <2 months postpartum)",
        "excl_reason": "Reason excluded from analysis sample", "pregnant": "Currently pregnant",
        "postpartum_lt2m": "Gave birth less than 2 months before interview",
        "bmi": "Body mass index (kg/m2)", "weight_kg": "Weight (kg)", "height_cm": "Height (cm)",
        "age": "Age (completed years)", "age_months": "Age (completed months)", "age_group": "Age group (years)",
        "who_ref": "Younger than 19 years: WHO 2007 BMI-for-age applies",
        "bmi_z": "BMI-for-age z-score, WHO 2007 reference (age < 19 years)",
        "cut_thin_A": "Thinness cut-off, definition A (kg/m2)", "cut_over_A": "Overweight cut-off, definition A (kg/m2)",
        "cut_obese_A": "Obesity cut-off, definition A (kg/m2)",
        "cut_thin_B": "Thinness cut-off, definition B: age-appropriate (kg/m2)",
        "cut_over_B": "Overweight cut-off, definition B: age-appropriate (kg/m2)",
        "cut_obese_B": "Obesity cut-off, definition B: age-appropriate (kg/m2)",
        "thin_A": "Thin, definition A (BMI < 18.5)", "over_A": "Overweight or obese, definition A (BMI >= 25)",
        "obese_A": "Obese, definition A (BMI >= 30)",
        "thin_B": "Thin, definition B (BMI-for-age z < -2 if < 19 y)",
        "over_B": "Overweight or obese, definition B (z > +1 if < 19 y)",
        "obese_B": "Obese, definition B (z > +2 if < 19 y)",
        "education": "Highest educational level", "educ_years": "Years of education",
        "wealth": "Wealth quintile", "wealth_score": "Wealth index factor score", "religion": "Religion",
        "marital": "Marital status", "parity": "Children ever born", "parity_group": "Children ever born (grouped)",
        "breastfeeding": "Currently breastfeeding", "hh_size": "Number of household members",
        "female_head": "Female head of household", "working": "Currently working",
        "media_weekly": "Reads, listens to radio or watches TV at least weekly",
        "modern_contra": "Currently uses a modern contraceptive method", "tobacco": "Uses any tobacco",
        "water": "Drinking water source (JMP)", "sanitation": "Sanitation facility (JMP)",
        "clean_fuel": "Cooking fuel",
    }
    YN = {0: "No", 1: "Yes"}
    value_labels = {
        "region": {int(k): v.title() for k, v in VL["v024"].items()},
        "residence": {1: "Urban", 2: "Rural"},
        "zone": {int(r.zone): f"{r.adm2_name} ({r.adm1_name})"[:80] for r in zones.itertuples()},
        "zone_assign": {0: "Within zone polygon", 1: "Nearest zone in survey region"},
        "stratum_design": {int(k): v.title() for k, v in VL["v023"].items()},
        "excl_reason": {0: "Included", 1: "Pregnant", 2: "Less than 2 months postpartum"},
        "age_group": {1: "15-19", 2: "20-24", 3: "25-29", 4: "30-34", 5: "35-39", 6: "40-44", 7: "45-49"},
        "education": {0: "No education", 1: "Primary", 2: "Secondary", 3: "Higher"},
        "wealth": {1: "Poorest", 2: "Poorer", 3: "Middle", 4: "Richer", 5: "Richest"},
        "religion": {1: "Orthodox", 2: "Protestant", 3: "Muslim", 4: "Other"},
        "marital": {1: "Never in union", 2: "Married or living together", 3: "Widowed, divorced or separated"},
        "parity_group": {1: "0", 2: "1-2", 3: "3-4", 4: "5 or more"},
        "water": {1: "Improved", 2: "Unimproved"},
        "sanitation": {1: "Improved", 2: "Unimproved", 3: "No facility"},
        "clean_fuel": {0: "Polluting fuel", 1: "Clean fuel", 2: "No cooking in household"},
    }
    for v in ["in_sample", "pregnant", "postpartum_lt2m", "who_ref", "thin_A", "over_A", "obese_A", "thin_B",
              "over_B", "obese_B", "breastfeeding", "female_head", "working", "media_weekly", "modern_contra",
              "tobacco"]:
        value_labels[v] = YN
    out.to_stata(OUT, version=118, write_index=False, variable_labels=VAR_LABELS, value_labels=value_labels,
                 data_label="Women's BMI, Ethiopia DHS 2024/25: analysis file")
    print(f"wrote {OUT}: {len(out)} women, {out['in_sample'].sum()} in analysis sample")


def load_application(root="."):
    """Analysis sample, location and scale design matrices, zone graph and indices."""
    import pyreadstat
    P = _paths(root)
    df, _ = pyreadstat.read_dta(os.path.join(P["app"], "ethiopia_women_bmi_2024.dta"))
    d = df[df["in_sample"] == 1].reset_index(drop=True)
    W = np.load(os.path.join(P["app"], "W_zones.npy"))
    z = lambda x: (x - x.mean()) / x.std()
    cols = {"Age (per SD)": z(d["age"]), "Age squared (per SD)": z(d["age"] ** 2),
            "Altitude (per SD)": z(d["altitude_m"]), "Children ever born (per SD)": z(d["parity"]),
            "Household size (per SD)": z(d["hh_size"]), "Rural residence": (d["residence"] == 2).astype(float)}
    for k, lab in [(1, "Primary"), (2, "Secondary"), (3, "Higher")]:
        cols[f"Education: {lab}"] = (d["education"] == k).astype(float)
    for k, lab in [(2, "Poorer"), (3, "Middle"), (4, "Richer"), (5, "Richest")]:
        cols[f"Wealth: {lab}"] = (d["wealth"] == k).astype(float)
    cols["Married or living together"] = (d["marital"] == 2).astype(float)
    cols["Widowed, divorced or separated"] = (d["marital"] == 3).astype(float)
    cols["Currently breastfeeding"] = d["breastfeeding"].astype(float)
    cols["Female household head"] = d["female_head"].astype(float)
    cols["Currently working"] = d["working"].astype(float)
    cols["Weekly media exposure"] = d["media_weekly"].astype(float)
    cols["Modern contraceptive use"] = d["modern_contra"].astype(float)
    cols["Uses tobacco"] = d["tobacco"].astype(float)
    cols["Unimproved water"] = (d["water"] == 2).astype(float)
    cols["Unimproved sanitation"] = (d["sanitation"] == 2).astype(float)
    cols["No sanitation facility"] = (d["sanitation"] == 3).astype(float)
    cols["Polluting cooking fuel"] = (d["clean_fuel"] == 0).astype(float)
    cols["Not a usual resident"] = d["water"].isna().astype(float)
    X = pd.DataFrame(cols)
    zc = {f"Region: {REGION_NAMES[r]}": (d["region"] == r).astype(float) for r in sorted(REGION_NAMES) if r != 4}
    zc["Urban residence"] = (d["residence"] == 1).astype(float)
    Z = pd.DataFrame(zc)
    assert not X.isna().any().any(), X.columns[X.isna().any()].tolist()
    return dict(d=d, X=X, Z=Z, W=W, area=d["zone"].to_numpy(int) - 1, cluster=pd.factorize(d["cluster"])[0], P=P)


def fit_application(root=".", variant="hetero", n_iter=12000, burn=4000, chains=4):
    """Fit the location-scale ('hetero') or location-only ('homo') model; saves draws and a summary."""
    A = load_application(root)
    d, X, Z, W, P = A["d"], A["X"], A["Z"], A["W"], A["P"]
    mod = SpatialRankLikelihood(d["bmi"].to_numpy(float), X.to_numpy(float), A["area"], A["cluster"], W,
                                Z_scale=Z.to_numpy(float) if variant == "hetero" else None)
    t0 = time.time()
    out = mod.fit(n_iter=n_iter, burn=burn, n_chains=chains, seed=2024, verbose=True)
    secs = time.time() - t0
    np.savez_compressed(os.path.join(P["app"], f"fit_srl_{variant}.npz"),
                        **{k: v for k, v in out.items() if k != "acceptance"},
                        covariates=np.array(X.columns), scale_covariates=np.array(Z.columns), seconds=secs)
    rows = summarise(out["beta"], list(X.columns))
    if variant == "hetero":
        rows += summarise(out["gamma"], [f"log-sd: {c}" for c in Z.columns])
    for k in ["sigma_phi", "rho", "sigma_eta"]:
        rows += summarise(out[k], [k])
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(P["app"], f"fit_srl_{variant}_summary.csv"), index=False)
    print(tab.round(3).to_string(index=False))
    print(f"acceptance: { {k: np.round(v, 2).tolist() for k, v in out['acceptance'].items()} }")
    print(f"run time {secs / 60:.1f} min")


def prevalence_application(root=".", variant="hetero", n_draws=400):
    """Zone prevalence of thinness and overweight under definitions A and B and their difference,
    from one fit. Zones without sampled women are predicted from the covariate profile of women in
    the same region, with the cluster effect integrated out."""
    A = load_application(root)
    d, X, Z, W, P, area, cluster = A["d"], A["X"], A["Z"], A["W"], A["P"], A["area"], A["cluster"]
    fit = np.load(os.path.join(P["app"], f"fit_srl_{variant}.npz"), allow_pickle=True)
    het = variant == "hetero"
    mod = SpatialRankLikelihood(d["bmi"].to_numpy(float), X.to_numpy(float), area, cluster, W,
                                Z_scale=Z.to_numpy(float) if het else None)
    gam = fit["gamma"].reshape(-1, Z.shape[1]) if het else None
    m = W.shape[0]
    y, w = mod.y, d["wt"].to_numpy(float)
    cuts = {k: d[k].to_numpy(float) for k in ["cut_thin_A", "cut_thin_B", "cut_over_A", "cut_over_B"]}
    grid = np.unique(np.round(np.concatenate(list(cuts.values())), 2))
    beta = fit["beta"].reshape(-1, mod.p); phi = fit["phi"].reshape(-1, m)
    eta = fit["eta"].reshape(-1, mod.K); sig_eta = fit["sigma_eta"].reshape(-1)
    idx = np.linspace(0, beta.shape[0] - 1, n_draws).astype(int)
    rng = np.random.default_rng(7)

    zones = pd.read_csv(os.path.join(P["app"], "zones_lookup.csv"))
    has_data = np.bincount(area, minlength=m) > 0
    zone_region = d.groupby("zone")["region"].agg(lambda s: s.mode().iat[0])
    region_of_zone = {}
    for zz in zones["zone"]:
        if zz in zone_region.index:
            region_of_zone[zz] = zone_region[zz]
        else:
            nb = np.where(W[zz - 1] > 0)[0] + 1
            region_of_zone[zz] = pd.Series([zone_region[n] for n in nb if n in zone_region.index]).mode().iat[0]
    empty = [zz for zz in zones["zone"] if not has_data[zz - 1]]

    labels = ["thin_A", "thin_B", "over_A", "over_B"]
    out = {k: np.empty((len(idx), m)) for k in labels}
    nat = {k: np.empty(len(idx)) for k in labels}
    wsum = np.bincount(area, weights=w, minlength=m)
    for j, s in enumerate(idx):
        mu = beta[s] @ mod.X.T + phi[s][area] + eta[s][cluster]
        sd = np.exp(mod.Zs @ gam[s]) if het else np.ones(mod.N)
        om = bootstrap_weights(rng, mod.N, cluster)
        tau_g = latent_threshold(grid, y, mu, om, sd=sd)
        tau = {k: tau_g[np.searchsorted(grid, np.round(c, 2))] for k, c in cuts.items()}
        p = {"thin_A": special.ndtr((tau["cut_thin_A"] - mu) / sd), "thin_B": special.ndtr((tau["cut_thin_B"] - mu) / sd),
             "over_A": special.ndtr((mu - tau["cut_over_A"]) / sd), "over_B": special.ndtr((mu - tau["cut_over_B"]) / sd)}
        for k in labels:
            out[k][j] = np.bincount(area, weights=w * p[k], minlength=m) / np.where(wsum > 0, wsum, 1)
            nat[k][j] = np.sum(w * p[k]) / w.sum()
        xb = beta[s] @ mod.X.T
        for zz in empty:
            r = d["region"].to_numpy() == region_of_zone[zz]
            sc = np.sqrt(sd[r] ** 2 + sig_eta[s] ** 2)
            mu_z = xb[r] + phi[s][zz - 1]
            out["thin_A"][j, zz - 1] = np.average(special.ndtr((tau["cut_thin_A"][r] - mu_z) / sc), weights=w[r])
            out["thin_B"][j, zz - 1] = np.average(special.ndtr((tau["cut_thin_B"][r] - mu_z) / sc), weights=w[r])
            out["over_A"][j, zz - 1] = np.average(special.ndtr((mu_z - tau["cut_over_A"][r]) / sc), weights=w[r])
            out["over_B"][j, zz - 1] = np.average(special.ndtr((mu_z - tau["cut_over_B"][r]) / sc), weights=w[r])
    for c in ["thin", "over"]:
        out[f"{c}_D"] = out[f"{c}_B"] - out[f"{c}_A"]
        nat[f"{c}_D"] = nat[f"{c}_B"] - nat[f"{c}_A"]

    res = zones.copy()
    res["has_data"] = has_data
    res["n_women"] = np.bincount(area, minlength=m)
    for k, v in out.items():
        res[f"{k}_mean"] = v.mean(0)
        res[f"{k}_lo"], res[f"{k}_hi"] = np.quantile(v, [0.025, 0.975], axis=0)
    res.to_csv(os.path.join(P["app"], f"zone_prevalence_{variant}.csv"), index=False)
    print("National (weighted, model-based): mean [95% CrI]")
    for k, v in nat.items():
        lo, hi = np.quantile(v, [0.025, 0.975])
        print(f"  {k:<7} {v.mean():.4f} [{lo:.4f}, {hi:.4f}]")
    reg = []
    for r, g in d.groupby("region"):
        sel = area[d["region"].to_numpy() == r]
        ww = w[d["region"].to_numpy() == r]
        row = {"region": int(r), "n": len(g)}
        for k in labels:
            model_r = np.array([np.average(out[k][j][sel], weights=ww) for j in range(len(idx))])
            row[f"{k}_model"] = model_r.mean()
            row[f"{k}_model_lo"], row[f"{k}_model_hi"] = np.quantile(model_r, [0.025, 0.975])
            row[f"{k}_direct"] = np.average(g[k], weights=g["wt"])
        reg.append(row)
    pd.DataFrame(reg).to_csv(os.path.join(P["app"], f"region_check_{variant}.csv"), index=False)


def maps_application(root=".", variant="hetero"):
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P = _paths(root)
    zones = gpd.read_file(P["admin"], layer="eth_admin2").sort_values("adm2_pcode").reset_index(drop=True)
    zones["zone"] = np.arange(1, len(zones) + 1)
    regions = gpd.read_file(P["admin"], layer="eth_admin1")
    res = pd.read_csv(os.path.join(P["app"], f"zone_prevalence_{variant}.csv"))
    g = zones.merge(res.drop(columns=["adm2_pcode", "adm2_name", "adm1_name"]), on="zone")
    for c in [c for c in g.columns if c.endswith("_mean")]:
        g[c] = 100 * g[c]
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    for k, name in [("thin", "fig_maps_thinness.png"), ("over", "fig_maps_overweight.png")]:
        fig, ax = plt.subplots(1, 3, figsize=(12, 4.6))
        lo = min(g[f"{k}_A_mean"].min(), g[f"{k}_B_mean"].min())
        hi = max(g[f"{k}_A_mean"].max(), g[f"{k}_B_mean"].max())
        panels = [(f"{k}_A_mean", "(a) Definition A (%)", lo, hi), (f"{k}_B_mean", "(b) Definition B (%)", lo, hi),
                  (f"{k}_D_mean", "(c) Difference B − A (percentage points)",
                   *({"thin": (-16.0, 0.0), "over": (0.0, max(1.0, g[f"{k}_D_mean"].max()))}[k]))]
        for a, (col, title, vmin, vmax) in zip(ax, panels):
            g.plot(column=col, cmap="viridis", vmin=vmin, vmax=vmax, ax=a, edgecolor="white", linewidth=0.2,
                   legend=True, legend_kwds={"shrink": 0.8, "orientation": "horizontal", "pad": 0.02,
                   **({"ticks": {"thin": [-16, -12, -8, -4, 0], "over": [0, 0.25, 0.5, 0.75, 1.0]}[k]} if col.endswith("_D_mean") else {})})
            g[~g["has_data"]].plot(ax=a, facecolor="none", edgecolor="black", hatch="///", linewidth=0.4)
            regions.boundary.plot(ax=a, color="black", linewidth=0.5)
            a.set_title(title, fontsize=10)
            a.set_axis_off()
        plt.tight_layout()
        fig.savefig(os.path.join(P["fig"], name), dpi=300, bbox_inches="tight")
        plt.close(fig)


def direct_estimate(g, k):
    """Weighted proportion and its linearised SE (clusters as PSUs, stratified)."""
    w = g["wt"].to_numpy(); y = g[k].to_numpy(float)
    p = np.average(y, weights=w)
    u = pd.DataFrame({"u": w * (y - p), "c": g["cluster"].to_numpy(), "h": g["strata"].to_numpy()})
    v = 0.0
    for _, gh in u.groupby("h"):
        t = gh.groupby("c")["u"].sum().to_numpy()
        nh = len(t)
        if nh > 1:
            v += nh / (nh - 1) * np.sum((t - t.mean()) ** 2)
    return p, np.sqrt(v) / w.sum()


def calibration_application(root=".", models=("homo", "hetero", "binary")):
    """Regional calibration (Table 5): direct estimates versus model-based estimates. The
    standardised difference uses the combined SE sqrt(SE_direct^2 + SD_model^2)."""
    A = load_application(root)
    d, P = A["d"], A["P"]
    rows = []
    for r, g in d.groupby("region"):
        row = {"region": REGION_NAMES[int(r)], "region_code": int(r), "n": len(g)}
        for k in ["thin_A", "over_A"]:
            row[f"{k}_direct"], row[f"{k}_se"] = direct_estimate(g, k)
        rows.append(row)
    tab = pd.DataFrame(rows)
    summ = []
    for v in models:
        path = os.path.join(P["app"], f"region_check_{v}.csv")
        if not os.path.exists(path):
            continue
        rc = pd.read_csv(path).rename(columns={"region": "region_code"})
        for k in ["thin_A", "over_A"]:
            m = tab.merge(rc[["region_code", f"{k}_model", f"{k}_model_lo", f"{k}_model_hi"]], on="region_code")
            tab[f"{k}_{v}"] = m[f"{k}_model"].to_numpy()
            diff = m[f"{k}_model"] - m[f"{k}_direct"]
            sd_model = (m[f"{k}_model_hi"] - m[f"{k}_model_lo"]) / 3.92
            zsc = diff / np.sqrt(m[f"{k}_se"] ** 2 + sd_model ** 2)
            summ.append(dict(model=v, outcome=k, mean_abs_diff_pp=100 * np.abs(diff).mean(),
                             within_2se=int((np.abs(zsc) < 2).sum()), sum_sq_std=float((zsc ** 2).sum())))
    tab.to_csv(os.path.join(P["tab"], "region_calibration.csv"), index=False)
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(P["tab"], "calibration_summary.csv"), index=False)
    print(summ.round(2).to_string(index=False))


def binomial_application(root=".", n_iter=12000, burn=4000, chains=4, n_draws=400):
    """Binary probit BYM2 with cluster effects for thinness and overweight under definition A
    (same covariates, zone graph and priors as the rank-likelihood model)."""
    A = load_application(root)
    d, X, W, P, area, cluster = A["d"], A["X"], A["W"], A["P"], A["area"], A["cluster"]
    w = d["wt"].to_numpy(float); Xm = X.to_numpy(float)
    region_rows, zone_rows = {}, {}
    for cond in ["thin", "over"]:
        ind = d[f"{cond}_A"].to_numpy(float)
        yb = 1.0 - ind if cond == "thin" else ind
        mod = OrdinalProbitBYM2(yb, Xm, area, cluster, W)
        out = mod.fit(n_iter=n_iter, burn=burn, n_chains=chains, seed=2024)
        rows = summarise(out["beta"], list(X.columns))
        for k in ["sigma_phi", "rho", "sigma_eta"]:
            rows += summarise(out[k], [k])
        pd.DataFrame(rows).to_csv(os.path.join(P["app"], f"fit_binary_{cond}_A_summary.csv"), index=False)
        S = out["beta"].shape[0] * out["beta"].shape[1]
        idx = np.linspace(0, S - 1, n_draws).astype(int)
        beta = out["beta"].reshape(S, -1)[idx]; phi = out["phi"].reshape(S, -1)[idx]
        eta = out["eta"].reshape(S, -1)[idx]; thr = out["thr"].reshape(S, -1)[idx, 0]
        mu = beta @ (Xm - Xm.mean(0)).T + phi[:, area] + eta[:, cluster]
        plow = special.ndtr(thr[:, None] - mu)
        p = plow if cond == "thin" else 1.0 - plow
        for key, rows_out in [("region", region_rows), ("zone", zone_rows)]:
            gg = d[key].to_numpy(int)
            for u in np.unique(gg):
                s = gg == u
                v = (p[:, s] * w[s]).sum(1) / w[s].sum()
                r = rows_out.setdefault(int(u), {key: int(u), "n": int(s.sum())})
                r[f"{cond}_A_model"] = v.mean()
                r[f"{cond}_A_model_lo"], r[f"{cond}_A_model_hi"] = np.quantile(v, [0.025, 0.975])
    pd.DataFrame(region_rows.values()).to_csv(os.path.join(P["app"], "region_check_binary.csv"), index=False)
    pd.DataFrame(zone_rows.values()).to_csv(os.path.join(P["app"], "zone_prevalence_binary.csv"), index=False)


def covariate_effects_application(root=".", n_draws=400):
    """Average marginal effects on the prevalence of thinness and overweight (adult cut-offs)."""
    A = load_application(root)
    d, X, Z, P, area, cluster = A["d"], A["X"], A["Z"], A["P"], A["area"], A["cluster"]
    fit = np.load(os.path.join(P["app"], "fit_srl_hetero.npz"), allow_pickle=True)
    names = list(X.columns)
    Xr = X.to_numpy(float); Xc = Xr - Xr.mean(0); Zs = Z.to_numpy(float)
    iu = list(Z.columns).index("Urban residence")
    y, w = d["bmi"].to_numpy(float), d["wt"].to_numpy(float)
    beta = fit["beta"].reshape(-1, len(names)); gam = fit["gamma"].reshape(-1, Zs.shape[1])
    phi = fit["phi"].reshape(-1, fit["phi"].shape[2]); eta = fit["eta"].reshape(-1, fit["eta"].shape[2])
    idx = np.linspace(0, beta.shape[0] - 1, n_draws).astype(int)
    groups = {"Education": ["Education: Primary", "Education: Secondary", "Education: Higher"],
              "Wealth": ["Wealth: Poorer", "Wealth: Middle", "Wealth: Richer", "Wealth: Richest"],
              "Marital": ["Married or living together", "Widowed, divorced or separated"],
              "Sanitation": ["Unimproved sanitation", "No sanitation facility"]}
    group_of = {c: g for g, cs in groups.items() for c in cs}
    cont = ["Altitude (per SD)", "Children ever born (per SD)", "Household size (per SD)"]

    def set_level(col, value):
        Xn = Xr.copy(); Zn = Zs.copy()
        if col in group_of:
            for c in groups[group_of[col]]:
                Xn[:, names.index(c)] = 0.0
        Xn[:, names.index(col)] = value
        if col == "Rural residence":
            Zn[:, iu] = 1.0 - value
        return Xn, Zn

    def set_age(a):
        Xn = Xr.copy(); age = d["age"].to_numpy(float)
        Xn[:, names.index("Age (per SD)")] = (a - age.mean()) / age.std()
        Xn[:, names.index("Age squared (per SD)")] = (a ** 2 - (age ** 2).mean()) / (age ** 2).std()
        return Xn, Zs

    contrasts = [("Age 35 versus 20 years", set_age(35.0), set_age(20.0))]
    for c in cont:
        Xn = Xr.copy(); Xn[:, names.index(c)] += 1.0
        contrasts.append((c.replace(" (per SD)", " (+1 SD)"), (Xn, Zs), (Xr, Zs)))
    for c in names:
        if not (c.startswith("Age") or c in cont):
            contrasts.append((c, set_level(c, 1.0), set_level(c, 0.0)))
    rng = np.random.default_rng(23)
    ame = {lab: {"thin": [], "over": []} for lab, *_ in contrasts}
    for s in idx:
        b, g = beta[s], gam[s]
        re = phi[s][area] + eta[s][cluster]
        mu = Xc @ b + re; sd = np.exp(Zs @ g)
        om = bootstrap_weights(rng, len(y), cluster)
        t_thin, t_over = latent_threshold(np.array([18.5, 25.0]), y, mu, om, sd=sd)
        for lab, (X1, Z1), (X0, Z0) in contrasts:
            mu1 = (X1 - Xr.mean(0)) @ b + re; mu0 = (X0 - Xr.mean(0)) @ b + re
            s1 = np.exp(Z1 @ g); s0 = np.exp(Z0 @ g)
            ame[lab]["thin"].append(100 * np.average(special.ndtr((t_thin - mu1) / s1) - special.ndtr((t_thin - mu0) / s0), weights=w))
            ame[lab]["over"].append(100 * np.average(special.ndtr((mu1 - t_over) / s1) - special.ndtr((mu0 - t_over) / s0), weights=w))
    rows = []
    for lab, *_ in contrasts:
        r = {"term": lab}
        for k in ["thin", "over"]:
            v = np.array(ame[lab][k])
            r[f"{k}_mean"], (r[f"{k}_lo"], r[f"{k}_hi"]) = v.mean(), np.quantile(v, [0.025, 0.975])
        rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(P["app"], "covariate_effects.csv"), index=False)
    print(res.round(2).to_string(index=False))



def scale_coefficients_application(root="."):
    """Scale coefficients of the location-scale model: zeta and the ratio of latent standard
    deviations exp(zeta) relative to rural women in Oromia."""
    P = _paths(root)
    fit = np.load(os.path.join(P["app"], "fit_srl_hetero.npz"), allow_pickle=True)
    gam = fit["gamma"].reshape(-1, fit["gamma"].shape[-1])
    names = [str(c).replace("Region: ", "") for c in fit["scale_covariates"]]
    lo, hi = np.quantile(gam, [0.025, 0.975], axis=0)
    tab = pd.DataFrame({"term": names, "zeta": gam.mean(0), "zeta_lo": lo, "zeta_hi": hi,
                        "sd_ratio": np.exp(gam).mean(0), "sd_ratio_lo": np.exp(lo), "sd_ratio_hi": np.exp(hi)})
    tab.to_csv(os.path.join(P["tab"], "scale_coefficients.csv"), index=False)
    print(tab.round(2).to_string(index=False))
    return tab


def zone_direct_application(root="."):
    """Design-based standard errors of zone-level direct estimates (why model-based estimates are needed)."""
    A = load_application(root)
    d, W, P = A["d"], A["W"], A["P"]
    rows = []
    for zz, g in d.groupby("zone"):
        p, se = direct_estimate(g, "thin_A")
        p2, se2 = direct_estimate(g, "over_A")
        rows.append(dict(zone=int(zz), n=len(g), thin_A=p, thin_A_se=se, over_A=p2, over_A_se=se2))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(P["tab"], "zone_direct_estimates.csv"), index=False)
    print(f"zones with data: {len(tab)} of {W.shape[0]}; women per zone: median {tab.n.median():.0f}, "
          f"range {tab.n.min()}-{tab.n.max()}")
    print(f"thinness SE (percentage points): median {100 * tab.thin_A_se.median():.2f}, "
          f"max {100 * tab.thin_A_se.max():.2f}; zones with SE > 5: {(tab.thin_A_se > 0.05).sum()}")
    print(f"median 95% CI width: {100 * 2 * 1.96 * tab.thin_A_se.median():.2f} percentage points")
    return tab


def compare_binomial_application(root="."):
    """Zone-level agreement and interval widths: rank-likelihood (location-scale) versus binomial BYM2."""
    P = _paths(root)
    zb = pd.read_csv(os.path.join(P["app"], "zone_prevalence_binary.csv"))
    zh = pd.read_csv(os.path.join(P["app"], "zone_prevalence_hetero.csv"))
    m = zh.merge(zb, on="zone")
    rows = []
    for k in ["thin_A", "over_A"]:
        a, b = m[f"{k}_mean"], m[f"{k}_model"]
        wl, wb = m[f"{k}_hi"] - m[f"{k}_lo"], m[f"{k}_model_hi"] - m[f"{k}_model_lo"]
        rows.append(dict(outcome=k, zones=len(m), correlation=np.corrcoef(a, b)[0, 1],
                         mean_abs_diff_pp=100 * np.abs(a - b).mean(), max_abs_diff_pp=100 * np.abs(a - b).max(),
                         width_rank_pp=100 * wl.mean(), width_binomial_pp=100 * wb.mean(),
                         median_width_ratio=(wl / wb).median(), zones_narrower=int((wl < wb).sum())))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(P["tab"], "binomial_comparison.csv"), index=False)
    print(tab.round(2).to_string(index=False))
    return tab


def _acf(x, max_lag):
    x = x - x.mean()
    f = np.fft.rfft(x, n=2 * len(x))
    ac = np.fft.irfft(f * np.conj(f))[:max_lag + 1]
    return ac / ac[0]


def diagnostics_application(root=".", variant="hetero"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    from matplotlib import cm
    P = _paths(root)
    fit = np.load(os.path.join(P["app"], f"fit_srl_{variant}.npz"), allow_pickle=True)
    cov = list(fit["covariates"]); scov = [str(c).replace("Region: ", "") for c in fit["scale_covariates"]]
    params = [(fit["sigma_phi"], r"$\sigma_\phi$"), (fit["rho"], r"$\rho$"), (fit["sigma_eta"], r"$\sigma_\eta$"),
              (fit["beta"][:, :, cov.index("Wealth: Richest")], r"$\beta$: richest wealth quintile"),
              (fit["gamma"][:, :, scov.index("Somali")], r"$\zeta$: Somali"),
              (fit["gamma"][:, :, scov.index("Urban residence")], r"$\zeta$: urban residence")]
    quantities = ([fit["beta"][:, :, j] for j in range(fit["beta"].shape[2])]
                  + [fit["gamma"][:, :, j] for j in range(fit["gamma"].shape[2])]
                  + [fit["phi"][:, :, j] for j in range(fit["phi"].shape[2])]
                  + [fit["sigma_phi"], fit["rho"], fit["sigma_eta"]])
    R = np.array([rhat(x) for x in quantities]); EB = np.array([ess_bulk(x) for x in quantities])
    ET = np.array([ess_tail(x) for x in quantities])
    print(f"{len(R)} quantities: max R-hat {R.max():.3f}, min bulk ESS {EB.min():.0f}, min tail ESS {ET.min():.0f}")
    cols = [cm.viridis(v) for v in [0.0, 0.33, 0.62, 0.88]]
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(3, 3, figsize=(7.5, 6.6))
    n_chain, n_draw = params[0][0].shape
    it = np.arange(n_draw)[::8]
    for k, (x, lab) in enumerate(params):
        a = ax[k // 3, k % 3]
        for c in range(n_chain):
            a.plot(it, x[c, ::8], color=cols[c], linewidth=0.35, alpha=0.8, label=f"Chain {c + 1}")
        a.set_title(lab, fontsize=8); a.set_xlabel("Iteration after burn-in")
    for a, (vals, lab, ref) in zip(ax[2], [(R, r"Split $\widehat{R}$", 1.01), (EB, "Bulk ESS", 400), (ET, "Tail ESS", 400)]):
        log = lab != r"Split $\widehat{R}$"
        bins = np.geomspace(min(vals.min(), ref) * 0.8, vals.max() * 1.2, 30) if log else np.linspace(1.0, 1.01, 31)
        a.hist(vals, bins=bins, color=cm.viridis(0.45), edgecolor="white", linewidth=0.4)
        a.axvline(ref, color="0.3", linestyle="--", linewidth=0.8)
        a.set_xlabel(lab); a.set_ylabel("Number of quantities")
        if log:
            a.set_xscale("log")
            a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    for a in ax.ravel():
        for sp in ["top", "right"]:
            a.spines[sp].set_visible(False)
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(P["fig"], "fig_diagnostics.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(4, 3, figsize=(7.5, 8.2))
    for k, (x, lab) in enumerate(params):
        a_d, a_a = ax[2 * (k // 3), k % 3], ax[2 * (k // 3) + 1, k % 3]
        lo, hi = np.quantile(x, [0.0005, 0.9995])
        grid = np.linspace(lo, hi, 300)
        for c in range(x.shape[0]):
            a_d.plot(grid, stats.gaussian_kde(x[c])(grid), color=cols[c], linewidth=1.0, label=f"Chain {c + 1}")
            a_a.plot(np.arange(41), _acf(x[c], 40), color=cols[c], linewidth=1.0)
        a_d.set_title(lab, fontsize=8); a_d.set_yticks([])
        a_a.axhline(0, color="0.6", linewidth=0.6); a_a.set_ylim(-0.1, 1.02); a_a.set_xlabel("Lag")
    for a in ax.ravel():
        for sp in ["top", "right"]:
            a.spines[sp].set_visible(False)
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(os.path.join(P["fig"], "fig_density_acf.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def ppc_application(root=".", n_rep=400):
    """Rank-based posterior predictive check of regional BMI distributions, location-only versus
    location-scale. For each draw, latent values are generated and the observed BMI values are
    assigned to them in rank order; observed statistics carry cluster-bootstrap 95% intervals."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    A = load_application(root)
    d, X, Z, P, area, cluster = A["d"], A["X"], A["Z"], A["P"], A["area"], A["cluster"]
    y = d["bmi"].to_numpy(float); ys = np.sort(y)
    region = d["region"].to_numpy(int)
    regs = sorted(np.unique(region), key=lambda r: REGION_NAMES[r])
    Xc = X.to_numpy(float); Xc = Xc - Xc.mean(0); Zs = Z.to_numpy(float)

    def stats_by_region(v):
        out = np.empty((len(regs), 4))
        for j, r in enumerate(regs):
            s = v[region == r]
            q1, q2, q3 = np.quantile(s, [0.25, 0.5, 0.75])
            out[j] = [q2, q3 - q1, 100 * np.mean(s < 18.5), 100 * np.mean(s >= 25.0)]
        return out

    obs = stats_by_region(y)
    clu = d["cluster"].to_numpy(); brng = np.random.default_rng(5)
    boot = np.empty((400, len(regs), 4))
    for j, r in enumerate(regs):
        s_idx = np.where(region == r)[0]
        cl = pd.Series(s_idx).groupby(clu[s_idx]).apply(np.array).tolist()
        for b in range(400):
            v = y[np.concatenate([cl[i] for i in brng.integers(0, len(cl), len(cl))])]
            q1, q2, q3 = np.quantile(v, [0.25, 0.5, 0.75])
            boot[b, j] = [q2, q3 - q1, 100 * np.mean(v < 18.5), 100 * np.mean(v >= 25.0)]
    obs_lo, obs_hi = np.quantile(boot, [0.025, 0.975], axis=0)

    rep, rng = {}, np.random.default_rng(11)
    for v in ["hetero", "homo"]:
        f = np.load(os.path.join(P["app"], f"fit_srl_{v}.npz"), allow_pickle=True)
        beta = f["beta"].reshape(-1, Xc.shape[1]); phi = f["phi"].reshape(-1, f["phi"].shape[2])
        eta = f["eta"].reshape(-1, f["eta"].shape[2])
        gam = f["gamma"].reshape(-1, Zs.shape[1]) if v == "hetero" else None
        idx = np.linspace(0, beta.shape[0] - 1, n_rep).astype(int)
        out = np.empty((n_rep, len(regs), 4))
        for j, s in enumerate(idx):
            mu = Xc @ beta[s] + phi[s][area] + eta[s][cluster]
            sd = np.exp(Zs @ gam[s]) if gam is not None else 1.0
            z = mu + sd * rng.standard_normal(len(y))
            yrep = np.empty_like(y); yrep[np.argsort(z, kind="stable")] = ys
            out[j] = stats_by_region(yrep)
        rep[v] = out

    labels = ["Median BMI (kg/m$^2$)", "Interquartile range (kg/m$^2$)", "BMI < 18.5 (%)", r"BMI $\geq$ 25 (%)"]
    mcol = {"homo": "#d62728", "hetero": "#1f77b4"}
    mlab = {"homo": "Location-only", "hetero": "Location-scale"}
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(1, 4, figsize=(7.5, 4.4), sharey=True)
    ypos = np.arange(len(regs))
    for k in range(4):
        a = ax[k]
        for off, v in [(-0.17, "homo"), (0.17, "hetero")]:
            lo, hi = np.quantile(rep[v][:, :, k], [0.025, 0.975], axis=0)
            a.hlines(ypos + off, lo, hi, color=mcol[v], linewidth=2.2, label=f"{mlab[v]} 95% interval")
            print(f"{labels[k]:<32} {mlab[v]:<15} regions inside: {int(((obs[:, k] >= lo) & (obs[:, k] <= hi)).sum())}")
        a.errorbar(obs[:, k], ypos, xerr=[obs[:, k] - obs_lo[:, k], obs_hi[:, k] - obs[:, k]], fmt="o",
                   color="#2ca02c", ms=3.5, elinewidth=0.9, capsize=1.5, zorder=3, label="Observed (95% interval)")
        a.set_xlabel(labels[k]); a.grid(axis="x", color="0.9", linewidth=0.5)
        for sp in ["top", "right"]:
            a.spines[sp].set_visible(False)
    ax[0].set_yticks(ypos); ax[0].set_yticklabels([REGION_NAMES[r] for r in regs]); ax[0].invert_yaxis()
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.55, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(P["fig"], "fig_ppc.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def adaptation_application(root=".", burn=4000):
    """Re-run the burn-in of the location-scale fit and record windowed acceptance rates."""
    A = load_application(root)
    d, X, Z, W, P = A["d"], A["X"], A["Z"], A["W"], A["P"]
    mod = SpatialRankLikelihood(d["bmi"].to_numpy(float), X.to_numpy(float), A["area"], A["cluster"], W,
                                Z_scale=Z.to_numpy(float))
    rows = []
    for ch, child in enumerate(np.random.SeedSequence(2024).spawn(4)):
        rng = np.random.default_rng(child)
        s = mod._init_state(rng, disperse=ch > 0)
        prop_sd = mod._init_prop_sd()
        hist = {k: [] for k in prop_sd}
        for it in range(burn):
            s, acc = mod._step(s, rng, True, prop_sd)
            for k, v in acc.items():
                hist[k].append(v)
            if (it + 1) % 50 == 0:
                for k in prop_sd:
                    if hist[k]:
                        rate = np.mean(np.asarray(hist[k][-50:], dtype=float), axis=0)
                        prop_sd[k] = prop_sd[k] * np.exp(rate - 0.40)
                        for j, r in enumerate(np.atleast_1d(rate)):
                            rows.append(dict(chain=ch + 1, block=k, comp=j, iter=it + 1, rate=float(r)))
    tr = pd.DataFrame(rows)
    tr["rate_200"] = tr.groupby(["block", "chain", "comp"])["rate"].transform(lambda x: x.rolling(4).mean())
    tr.to_csv(os.path.join(P["app"], "adaptation_trace.csv"), index=False)
    first = tr[tr["rate_200"].between(0.33, 0.47)].groupby(["block", "chain", "comp"])["iter"].min()
    print(f"200-iteration acceptance first within 0.33-0.47: median {first.median():.0f}, max {first.max():.0f} iterations")


def example(data_dir="data", seed=1):
    """Fit the model to one simulated dataset on the region graph and report prevalence."""
    W = np.load(os.path.join(data_dir, "W_regions.npy"))
    d = simulate(W, n_clusters_per_area=2, error="normal", outcome="continuous", seed=seed)
    mod = SpatialRankLikelihood(d["y"], d["X"], d["area"], d["cluster"], W)
    out = mod.fit(n_iter=1000, burn=400, n_chains=2, seed=seed)
    area_prev, overall = prevalence(mod, 11.0, n_draws=200, seed=seed)
    truth = np.bincount(d["area"], weights=true_prevalence(d, 11.0)) / np.bincount(d["area"])
    print("max R-hat (beta):", round(max(rhat(out["beta"][:, :, j]) for j in range(out["beta"].shape[2])), 3))
    print("overall prevalence: %.3f [%.3f, %.3f]" % (overall.mean(), *np.quantile(overall, [0.025, 0.975])))
    print(pd.DataFrame({"area": np.arange(1, W.shape[0] + 1), "estimate": area_prev.mean(0),
                        "truth": truth}).round(3).to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["example", "simulate", "summarise", "sim-outputs", "extract", "fit",
                                        "prevalence", "maps", "binomial", "calibration", "effects",
                                        "diagnostics", "ppc", "adaptation", "sim-numbers", "scale",
                                        "zone-direct", "compare-binomial"])
    ap.add_argument("--root", default=".", help="project folder containing data/ (default: current folder)")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--only", default=None, help="simulate only scenarios whose name contains this text")
    ap.add_argument("--variant", default="hetero", choices=["hetero", "homo"])
    ap.add_argument("--iter", type=int, default=12000)
    ap.add_argument("--burn", type=int, default=4000)
    ap.add_argument("--chains", type=int, default=4)
    a = ap.parse_args(argv)
    data_dir = os.path.join(a.root, "data")
    sim_dir = os.path.join(a.root, "sim_results")
    summary = os.path.join(a.root, "simulation_summary.csv")
    if a.command == "example":
        example(data_dir)
    elif a.command == "simulate":
        run_simulation(a.reps, a.jobs, sim_dir, data_dir, a.only)
    elif a.command == "summarise":
        summarise_simulation(sim_dir, summary)
    elif a.command == "sim-outputs":
        simulation_outputs(summary, os.path.join(a.root, "figures"), os.path.join(a.root, "tables"))
    elif a.command == "extract":
        extract_data(a.root)
    elif a.command == "fit":
        fit_application(a.root, a.variant, a.iter, a.burn, a.chains)
    elif a.command == "prevalence":
        prevalence_application(a.root, a.variant)
    elif a.command == "maps":
        maps_application(a.root, a.variant)
    elif a.command == "binomial":
        binomial_application(a.root, a.iter, a.burn, a.chains)
    elif a.command == "calibration":
        calibration_application(a.root)
    elif a.command == "effects":
        covariate_effects_application(a.root)
    elif a.command == "diagnostics":
        diagnostics_application(a.root, a.variant)
    elif a.command == "ppc":
        ppc_application(a.root)
    elif a.command == "adaptation":
        adaptation_application(a.root, a.burn)
    elif a.command == "sim-numbers":
        simulation_key_numbers(summary, os.path.join(a.root, "simulation_key_numbers.csv"))
    elif a.command == "scale":
        scale_coefficients_application(a.root)
    elif a.command == "zone-direct":
        zone_direct_application(a.root)
    elif a.command == "compare-binomial":
        compare_binomial_application(a.root)


if __name__ == "__main__":
    main()
