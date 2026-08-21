"""Stages 1-4 in order, for one trajectory."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .descriptors import find_domains
from .ood import load_model, score_domain
from .retrieval import retrieve


@dataclass
class DomainReport:
    index: int
    n_particles: int
    n_centres: int
    ls_ratio: float
    ood: object
    retrieval: object = None
    error: str = ""


@dataclass
class Report:
    path: str
    frame: int
    domains: list = field(default_factory=list)


def analyze(gsd_path, frame: int = -1, config: Config | None = None,
            min_centres: int = 64, top_k: int = 3) -> Report:
    cfg = config or Config.load()
    model = load_model()
    rep = Report(path=str(gsd_path), frame=frame)
    for dom in find_domains(gsd_path, frame, min_centres=min_centres):
        try:
            o = score_domain(dom, model)
        except Exception as exc:  # noqa: BLE001
            rep.domains.append(DomainReport(dom.index, dom.n_particles, dom.n_centres,
                                            dom.ls_ratio, None, None, f"{type(exc).__name__}: {exc}"))
            continue
        try:
            r = retrieve(dom, cfg, top_k=top_k)
        except Exception as exc:  # noqa: BLE001
            r, err = None, f"retrieval failed: {type(exc).__name__}: {exc}"
        else:
            err = ""
        rep.domains.append(DomainReport(dom.index, dom.n_particles, dom.n_centres,
                                        dom.ls_ratio, o, r, err))
    return rep
