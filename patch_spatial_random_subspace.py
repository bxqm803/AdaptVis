#!/usr/bin/env python3
"""Extend AdaptVis's existing gradient/span script with dimension-matched
Haar-random subspace projections. No repository commit is required.

Run this in /workspace/AdaptVis:
    python - <<'PYCODE'
    ... paste this entire file's contents ...
    PYCODE
Or download this file into RunPod, then:
    python patch_spatial_random_subspace.py
"""
from pathlib import Path
import ast

source = Path("eval_decision_gradient_vs_relation_span_multilayer_v1.py")
target = Path("eval_decision_gradient_vs_relation_span_random_control_v2.py")
if not source.is_file():
    raise SystemExit(f"Missing original script: {source.resolve()}")
if target.exists():
    raise SystemExit(f"Refusing to overwrite existing script: {target.resolve()}")
s = source.read_text(encoding="utf-8")

def replace_exact(old: str, new: str):
    global s
    n = s.count(old)
    if n != 1:
        raise RuntimeError(f"Expected exactly 1 occurrence; found {n}: {old[:100]!r}")
    s = s.replace(old, new, 1)

replace_exact(
    '    p.add_argument("--random-repeats", type=int, default=20)\n',
    '    p.add_argument("--random-repeats", type=int, default=20)\n'
    '    p.add_argument("--random-subspaces", type=int, default=128, '\
    'help="Haar-random orthonormal subspaces per layer (same rank as spatial subspace)")\n',
)

new_helpers = '''

def make_dimension_matched_random_bases(Qsp_by_layer, num_draws: int, seed: int):
    """Sample once per layer and reuse SAME random bases across samples/tokens.

    For Q_sp in R^{D x k}, generate R Haar-random k-dimensional subspaces
    from QR of i.i.d. standard Gaussian matrices in R^{D x k}.
    """
    if num_draws < 2:
        raise ValueError("--random-subspaces must be >= 2")
    packed = {}
    geometry = {}
    for L in sorted(Qsp_by_layer):
        D, rank = map(int, Qsp_by_layer[L].shape)
        if not 0 < rank < D:
            raise RuntimeError(f"L{L}: invalid basis shape {(D, rank)}")
        rng = np.random.default_rng(seed + 10007 * int(L))
        mats = []
        for _ in range(num_draws):
            a = rng.standard_normal((D, rank))
            q, _ = np.linalg.qr(a, mode="reduced")
            mats.append(q.astype(np.float32))
        packed[int(L)] = np.concatenate(mats, axis=1)
        geometry[int(L)] = {"rank":rank, "hidden_size":D, "random_expectation":rank/D}
    return packed, geometry


def store_random_overlap(g, random_pack, draws, rank, cache, sid, layer, position):
    """Compare the SAME gradient g to each fixed random subspace at its layer."""
    g = np.asarray(g, dtype=np.float32)
    den = float(np.dot(g, g))
    if den <= EPS or not np.isfinite(den):
        cache[(int(sid), int(layer), int(position))] = np.full(draws, np.nan, np.float32)
        return float("nan")
    qtg = g @ random_pack
    null = (qtg.reshape(draws, rank).astype(np.float64) ** 2).sum(axis=1) / den
    cache[(int(sid), int(layer), int(position))] = null.astype(np.float32)
    return float(np.mean(null))


def summarize_dimension_matched_null(per_token, selection, cache, layers, geometry, draws, seed, outdir):
    """Evaluate original Top-K positions (selected at anchor L25) in every layer.

    The random basis draw index is SHARED across all samples and token positions
    in a given layer. This preserves correlations between gradients, unlike
    independent per-token Beta sampling.
    """
    top = selection[selection["selection"] == "top"]
    actual = {
        (int(r.sid), int(r.layer), int(r.position)): float(r.rho_grad_spatial)
        for r in per_token.itertuples(index=False)
    }
    sample_rows = []
    null_vectors = {}
    for (sid, k), group in top.groupby(["sid", "k"], sort=True):
        sid, k = int(sid), int(k)
        positions = group["position"].astype(int).tolist()
        for layer in layers:
            available = [
                p for p in positions
                if (sid, int(layer), p) in cache
                and (sid, int(layer), p) in actual
                and np.isfinite(actual[(sid, int(layer), p)])
                and np.isfinite(cache[(sid, int(layer), p)]).all()
            ]
            if not available:
                continue
            spatial = float(np.mean([actual[(sid, int(layer), p)] for p in available]))
            null_vector = np.stack(
                [cache[(sid, int(layer), p)] for p in available], axis=0
            ).mean(axis=0)
            null_vectors[(sid, int(layer), k)] = null_vector
            sample_rows.append({
                "sid":sid, "layer":int(layer), "k_tokens":k,
                "n_selected":len(available),
                "spatial_rho":spatial,
                "random_mean_rho":float(np.mean(null_vector)),
                "spatial_minus_random_mean":float(spatial-np.mean(null_vector)),
                "random_expectation":float(geometry[int(layer)]["random_expectation"]),
            })

    sample = pd.DataFrame(sample_rows)
    if sample.empty:
        raise RuntimeError("No Top-K / gradient rows matched; check anchor_selection.csv")
    sample.to_csv(outdir / "sample_topk_random_subspace.csv", index=False)

    rng = np.random.default_rng(seed + 654321)
    summary, null_draw_rows = [], []
    for (layer, k), group in sample.groupby(["layer", "k_tokens"], sort=True):
        layer, k = int(layer), int(k)
        sids = group["sid"].astype(int).tolist()
        null_matrix = np.stack([null_vectors[(sid, layer, k)] for sid in sids])
        null_group_means = np.mean(null_matrix, axis=0)
        spatial_values = group["spatial_rho"].to_numpy(np.float64)
        spatial_mean = float(np.mean(spatial_values))
        n = len(spatial_values)
        # Sample-level bootstrap, not token-level bootstrap.
        boot = np.mean(spatial_values[rng.integers(0, n, size=(1000, n))], axis=1)
        null_mean = float(null_group_means.mean())
        summary.append({
            "layer":layer, "k_tokens":k, "N_samples":n,
            "hidden_size":geometry[layer]["hidden_size"],
            "spatial_rank":geometry[layer]["rank"],
            "random_expectation":geometry[layer]["random_expectation"],
            "spatial_mean_rho":spatial_mean,
            "spatial_bootstrap_ci95_lo":float(np.percentile(boot,2.5)),
            "spatial_bootstrap_ci95_hi":float(np.percentile(boot,97.5)),
            "random_mean_rho":null_mean,
            "random_draw_ci95_lo":float(np.percentile(null_group_means,2.5)),
            "random_draw_ci95_hi":float(np.percentile(null_group_means,97.5)),
            "spatial_over_random_ratio":float(spatial_mean/null_mean) if null_mean > 0 else float("nan"),
            "spatial_minus_random":float(spatial_mean-null_mean),
            "p_random_ge_spatial":float((1 + np.count_nonzero(null_group_means >= spatial_mean)) / (draws + 1)),
            "p_random_le_spatial":float((1 + np.count_nonzero(null_group_means <= spatial_mean)) / (draws + 1)),
        })
        for rep, val in enumerate(null_group_means):
            null_draw_rows.append({
                "layer":layer,"k_tokens":k,"random_draw":rep,"mean_rho":float(val)
            })

    result = pd.DataFrame(summary)
    result.to_csv(outdir / "topk_random_subspace_summary.csv", index=False)
    pd.DataFrame(null_draw_rows).to_csv(outdir / "random_subspace_draws.csv", index=False)
    print("\\n" + "=" * 105)
    print("DIMENSION-MATCHED RANDOM SUBSPACE CONTROL (all values in PERCENT)")
    print("=" * 105)
    print("layer  TopK   N    rank/D      spatial       random       enrichment    p(rand>=spatial)")
    for r in result.itertuples(index=False):
        print(f"L{r.layer:<3}   {r.k_tokens:<4} {r.N_samples:4}   "
              f"{r.spatial_rank}/{r.hidden_size:<7}  "
              f"{100*r.spatial_mean_rho:8.4f}%    "
              f"{100*r.random_mean_rho:8.4f}%    "
              f"{r.spatial_over_random_ratio:8.2f}x       "
              f"{r.p_random_ge_spatial:.5f}")
    return result

'''
replace_exact('\ndef main():\n', new_helpers + '\ndef main():\n')

replace_exact(
    '        geom.to_csv(outdir / "spatial_basis_geometry.csv", index=False)\n',
    '        geom.to_csv(outdir / "spatial_basis_geometry.csv", index=False)\n'
    '        random_bases, random_geometry = make_dimension_matched_random_bases(\n'
    '            Qsp_by_layer, int(a.random_subspaces), int(a.seed) + 120003\n'
    '        )\n'
    '        random_projection_cache = {}\n'
    '        print("[RANDOM SUBSPACE]", int(a.random_subspaces), "Haar-random bases per layer; rank matched", flush=True)\n',
)

replace_exact(
    '                                "rho_grad_spatial": projection_ratio(g, Qsp),\n',
    '                                "rho_grad_spatial": projection_ratio(g, Qsp),\n'
    '                                "rho_grad_random_subspace_mean": store_random_overlap(\n'
    '                                    g, random_bases[L], int(a.random_subspaces),\n'
    '                                    int(Qsp.shape[1]), random_projection_cache,\n'
    '                                    sid, L, p,\n'
    '                                ),\n',
)

replace_exact(
    '        summary.to_csv(outdir / "topk_projection_summary.csv", index=False)\n',
    '        summary.to_csv(outdir / "topk_projection_summary.csv", index=False)\n'
    '        summarize_dimension_matched_null(\n'
    '            df, sel_df, random_projection_cache, layers, random_geometry,\n'
    '            int(a.random_subspaces), int(a.seed), outdir\n'
    '        )\n',
)

ast.parse(s, filename=str(target))
target.write_text(s, encoding="utf-8")
print(f"[OK] Generated: {target.resolve()}")
print("[CHECK] Run: python -m py_compile", target)
print("[NOTE] Original script unchanged; no GitHub upload needed.")
