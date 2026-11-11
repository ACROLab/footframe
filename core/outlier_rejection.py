"""L07 Outlier Rejection: median, MAD and multi-photo measurement fusion.

Independently processes the physical measurements produced by a OpenCV 2D
pipeline. Does NOT average raw photo pixels, force a target shoe size,
or use any calibration from a subject's known foot length.
"""
from dataclasses import dataclass,asdict
from typing import Sequence
import numpy as np


@dataclass(frozen=True)
class Observation:
    view_id: str
    kind: str         # 'length' or 'width'
    mm: float
    paper_valid: bool
    foot_valid: bool
    view_valid: bool
    geometric_error_px: float = 0.0
    rejection_reason: str | None = None


@dataclass
class FusionResult:
    accepted_mm: float | None
    status: str
    used_views: list[str]
    rejected: dict[str,str]
    spread_mm: float | None
    observations: dict[str,float]

    def as_dict(self):return asdict(self)


def fuse(observations: Sequence[Observation],*,min_consensus=2)->FusionResult:
    """Conservative fusion after independent A4, silhouette and pose gates.

    `min_consensus` is number of *valid independent photos*, never photo
    pixels. An inlier is compared against MAD and a fixed small numerical
    tolerance (3 mm) for nearly identical candidates, not any known foot size.
    The spread is an inter-observation disagreement, not a statistical CI.
    """
    if min_consensus<2:raise ValueError('at least two independently valid views required')
    rejects={};accepted=[]
    for o in observations:
        if not np.isfinite(o.mm) or o.mm<=0:
            rejects[o.view_id]=o.rejection_reason or 'nonfinite_or_nonpositive_measurement'
        elif not o.paper_valid:
            rejects[o.view_id]=o.rejection_reason or 'paper_not_validated'
        elif not o.foot_valid:
            rejects[o.view_id]=o.rejection_reason or 'foot_not_validated'
        elif not o.view_valid:
            rejects[o.view_id]=o.rejection_reason or 'view_not_validated'
        elif not np.isfinite(o.geometric_error_px) or o.geometric_error_px<0:
            rejects[o.view_id]='invalid_geometry_error'
        else:accepted.append(o)
    if len(accepted)<min_consensus:
        return FusionResult(None,'insufficient_valid_views',[],rejects,None,{o.view_id:o.mm for o in observations})
    kinds={o.kind for o in accepted}
    if len(kinds)!=1:
        raise ValueError('do not mix length/width; fuse separately')
    values=np.asarray([o.mm for o in accepted],float)
    centre=float(np.median(values))
    mad=float(np.median(np.abs(values-centre)))
    # With three observations and a small MAD, the old threshold (3 mm) could
    # discard a 4.5 mm difference while keeping two nearby biased views.
    # Require a material disagreement before calling a photo an outlier;
    # this threshold is a repeatability gate, NOT a reference foot dimension.
    threshold=max(10.,3.*1.4826*mad)
    candidates=[o for o in accepted if abs(o.mm-centre)<=threshold]
    for o in accepted:
        if o not in candidates:rejects[o.view_id]='inconsistent_with_other_views'
    if len(candidates)<min_consensus:
        return FusionResult(None,'no_stable_consensus',[],rejects,None,{o.view_id:o.mm for o in observations})
    # Reject an ambiguous two-cluster set even if MAD would be too permissive.
    # A 2-view group cannot validate an outlier; differences > 12 mm are not fused.
    arr=np.array([o.mm for o in candidates])
    if len(candidates)==2 and np.ptp(arr)>12:
        return FusionResult(None,'two_views_disagree',[],rejects,float(np.ptp(arr)),{o.view_id:o.mm for o in observations})
    # Weight geometrically well-validated observations, but cap its influence
    # so a single photo cannot dominate a consensus.
    errors=np.array([o.geometric_error_px for o in candidates],float)
    weights=np.clip(1./(1.+errors/2.0),0.25,1.)
    weights/=weights.sum()
    estimate=float(np.sum(weights*arr))
    spread=float(np.ptp(arr)) if len(arr)>1 else 0.
    return FusionResult(estimate,'cross_view_consistent' if len(candidates)>=3 and spread<=10 else 'provisional',
        [o.view_id for o in candidates],rejects,spread,{o.view_id:o.mm for o in observations})
