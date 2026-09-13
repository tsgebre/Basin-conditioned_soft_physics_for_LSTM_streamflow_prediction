"""Basin attributes -> per-constraint weights.

This is the paper's actual hypothesis in about forty lines. Every weight is a
fixed function of things you can look up about a catchment before training:

    s   snow fraction        0 = never snows, 1 = all precip falls as snow
    a   aridity (PET/P)      > 1 = dry
    h   humidity index       min(1, 1/a), so h -> 0 as the basin dries out
    mwc max water content    soil storage capacity, metres

Nothing here is learned and nothing sees the training targets. The weights are frozen before the first gradient step (that's what makes the comparison against the uniform arm fair).

The normalisation at the end conditioned and uniform
arms are forced to the same TOTAL weight budget, so they differ in how the budget is spread, not in how hard it pushes. Without that, "conditioning helps"
and "more regularisation helps" would be the same experiment.
"""

ACTIVE = ["trend_alignment", "temp_gated_trend", "melt_response",
          "antecedent_wetness", "adaptive_lag", "volume_plausibility",
          "nonneg_monotone"]

BUDGET = float(len(ACTIVE))    # weights always sum to this


def raw_weights(s, a, mwc, mwc_ref):
    """Unnormalised applicability of each constraint to this basin."""
    h = 1.0 if a <= 0 else min(1.0, 1.0 / a)
    return {
        # rain-driven response: matters in rainy, non-snowy basins
        "trend_alignment": (1.0 - s) * h,
        # mixed rain/snow: peaks at s = 0.5, irrelevant at both extremes
        "temp_gated_trend": 4.0 * s * (1.0 - s),
        # melt: scales straight with how much of the precip is snow
        "melt_response": s,
        # antecedent wetness: only informative if the soil can't just soak it up
        "antecedent_wetness": h * (1.0 - min(mwc / mwc_ref, 1.0)),
        # lag and non-negativity are basin-independent -- always fully on
        "adaptive_lag": 1.0,
        "nonneg_monotone": 1.0,
        # volume closure binds harder in dry basins, where runoff ratios are low
        "volume_plausibility": 0.5 + 0.25 * min(a, 2.0),
    }


def conditioned(s, a, mwc, mwc_ref):
    """Normalised to BUDGET. This is what conditioned_physics trains with."""
    raw = raw_weights(s, a, mwc, mwc_ref)
    total = sum(raw.values())
    return {c: BUDGET * v / total for c, v in raw.items()}


def uniform(names=None):
    """Equal weight, same budget. What uniform_physics trains with."""
    names = names or ACTIVE
    return {c: BUDGET / len(names) for c in names}


def weights_for(arm_weighting, names, attrs=None, mwc_ref=0.724):
    """Dispatch used by the trainer. `attrs` needs frac_snow/aridity/max_water_content."""
    if arm_weighting == "none":
        return {}
    if arm_weighting == "uniform":
        return uniform(names)
    if arm_weighting == "conditioned":
        s, a, mwc = attrs["frac_snow"], attrs["aridity"], attrs["max_water_content"]
        # generic arms carry one constraint; give it the whole budget, scaled
        # by the basin's mean applicability so it stays attribute-sensitive.
        # Use the RAW weights: the normalised ones always average to 1.
        if set(names) - set(ACTIVE):
            raw = raw_weights(s, a, mwc, mwc_ref)
            scale = sum(raw.values()) / len(raw)
            return {n: BUDGET * scale / len(names) for n in names}
        w = conditioned(s, a, mwc, mwc_ref)
        return {c: w[c] for c in names}
    raise ValueError(f"unknown weighting: {arm_weighting}")
