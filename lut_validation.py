import matplotlib
import numpy as np
from scipy import stats

MAPS = ("gray", "viridis", "plasma", "inferno", "magma", "cividis")


def table():
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Map & unique & coll. & src. rev. & 8-bit rev. & max drop & $\rho$ & max err. \\", r"\midrule"]
    luma = np.array([0.2126, 0.7152, 0.0722])

    def render(x, name):
        rgb = matplotlib.colormaps[name](x)[..., :3].astype(np.float32)
        return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)

    lut = {}
    for name in MAPS:
        t = np.linspace(0.0, 1.0, 256)
        lut8 = render(t, name)
        unique = len({tuple(v) for v in lut8.tolist()})
        source_reversals = int(np.sum(np.diff(matplotlib.colormaps[name](t)[:, :3] @ luma) < -1e-9))
        steps = np.diff((lut8 / 255.0) @ luma)
        reversals = int(np.sum(steps < -1e-9))
        drop = float(-steps.min()) if reversals else 0.0
        rho = float(stats.spearmanr(t, (lut8 / 255.0) @ luma).correlation)
        dense = np.linspace(0.0, 1.0, 1024)
        coded = render(dense, name).astype(np.int32)
        reference = lut8.astype(np.int32)
        err = max(abs(int(np.argmin(((reference - c) ** 2).sum(axis=1))) / 255.0 - d) for c, d in zip(coded, dense))
        lut[name] = {"unique": unique, "reversals": reversals, "max_drop": drop, "rho": rho}
        lines.append(f"{name} & {unique} & {256 - unique} & {source_reversals} & {reversals} & {drop:.4f} & {rho:.3f} & {err:.4f} \\\\")
    return lines + [r"\bottomrule", r"\end{tabular}"], lut
