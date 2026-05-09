#!/usr/bin/env python3
"""Reference parser for the POUNCE iterate-trace format (POUNCEIT v1).

See FORMAT.md for the full specification. This script reads a dump file
produced by Ipopt patched with the pounce-iter-dump branch (or by the
POUNCE Rust port) and prints a structured summary.

Usage:
    python3 inspect.py <path-to-dump-file> [--full]
"""

from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

MAGIC = b"POUNCEIT"


@dataclass
class Header:
    format_version: int
    n: int
    m: int
    nnz_jac: int
    nnz_h: int
    name: str


@dataclass
class IterRecord:
    iter: int
    status: int
    mu: float
    tau: float
    alpha_pr: float
    alpha_du: float
    delta_x: float
    delta_s: float
    delta_c: float
    delta_d: float
    inf_pr: float
    inf_du: float
    constr_viol: float
    dual_inf: float
    complementarity: float
    f: float
    x: List[float]
    s: List[float]
    y_c: List[float]
    y_d: List[float]
    z_L: List[float]
    z_U: List[float]
    v_L: List[float]
    v_U: List[float]
    filter: List[Tuple[float, float]] = field(default_factory=list)


def _read_exact(buf: memoryview, offset: int, n: int) -> Tuple[memoryview, int]:
    if offset + n > len(buf):
        raise ValueError(
            f"truncated: wanted {n} bytes at offset {offset}, "
            f"file is {len(buf)} bytes"
        )
    return buf[offset : offset + n], offset + n


def _read_u32(buf: memoryview, offset: int) -> Tuple[int, int]:
    chunk, offset = _read_exact(buf, offset, 4)
    return struct.unpack_from("<I", chunk)[0], offset


def _read_f64(buf: memoryview, offset: int) -> Tuple[float, int]:
    chunk, offset = _read_exact(buf, offset, 8)
    return struct.unpack_from("<d", chunk)[0], offset


def _read_vec(buf: memoryview, offset: int) -> Tuple[List[float], int]:
    length, offset = _read_u32(buf, offset)
    if length == 0:
        return [], offset
    chunk, offset = _read_exact(buf, offset, 8 * length)
    return list(struct.unpack_from(f"<{length}d", chunk)), offset


def parse_header(buf: memoryview, offset: int = 0) -> Tuple[Header, int]:
    magic, offset = _read_exact(buf, offset, 8)
    if bytes(magic) != MAGIC:
        raise ValueError(f"bad magic: {bytes(magic)!r} (expected {MAGIC!r})")
    version, offset = _read_u32(buf, offset)
    if version != 1:
        raise ValueError(f"unsupported format_version {version} (only 1 known)")
    n, offset = _read_u32(buf, offset)
    m, offset = _read_u32(buf, offset)
    nnz_jac, offset = _read_u32(buf, offset)
    nnz_h, offset = _read_u32(buf, offset)
    name_len, offset = _read_u32(buf, offset)
    name_bytes, offset = _read_exact(buf, offset, name_len)
    name = bytes(name_bytes).decode("utf-8", errors="replace")
    return Header(version, n, m, nnz_jac, nnz_h, name), offset


def parse_record(buf: memoryview, offset: int) -> Tuple[IterRecord, int]:
    iter_idx, offset = _read_u32(buf, offset)
    status, offset = _read_u32(buf, offset)
    scalars: List[float] = []
    for _ in range(14):
        v, offset = _read_f64(buf, offset)
        scalars.append(v)
    (
        mu,
        tau,
        alpha_pr,
        alpha_du,
        delta_x,
        delta_s,
        delta_c,
        delta_d,
        inf_pr,
        inf_du,
        constr_viol,
        dual_inf,
        complementarity,
        f_val,
    ) = scalars
    x, offset = _read_vec(buf, offset)
    s, offset = _read_vec(buf, offset)
    y_c, offset = _read_vec(buf, offset)
    y_d, offset = _read_vec(buf, offset)
    z_L, offset = _read_vec(buf, offset)
    z_U, offset = _read_vec(buf, offset)
    v_L, offset = _read_vec(buf, offset)
    v_U, offset = _read_vec(buf, offset)
    filter_count, offset = _read_u32(buf, offset)
    filt: List[Tuple[float, float]] = []
    for _ in range(filter_count):
        theta, offset = _read_f64(buf, offset)
        phi, offset = _read_f64(buf, offset)
        filt.append((theta, phi))
    return (
        IterRecord(
            iter=iter_idx,
            status=status,
            mu=mu,
            tau=tau,
            alpha_pr=alpha_pr,
            alpha_du=alpha_du,
            delta_x=delta_x,
            delta_s=delta_s,
            delta_c=delta_c,
            delta_d=delta_d,
            inf_pr=inf_pr,
            inf_du=inf_du,
            constr_viol=constr_viol,
            dual_inf=dual_inf,
            complementarity=complementarity,
            f=f_val,
            x=x,
            s=s,
            y_c=y_c,
            y_d=y_d,
            z_L=z_L,
            z_U=z_U,
            v_L=v_L,
            v_U=v_U,
            filter=filt,
        ),
        offset,
    )


def parse_file(path: Path) -> Tuple[Header, List[IterRecord]]:
    data = path.read_bytes()
    buf = memoryview(data)
    header, offset = parse_header(buf, 0)
    records: List[IterRecord] = []
    while offset < len(buf):
        rec, offset = parse_record(buf, offset)
        records.append(rec)
    return header, records


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path)
    ap.add_argument(
        "--full",
        action="store_true",
        help="dump every vector entry (default: shapes + scalars only)",
    )
    args = ap.parse_args(argv)

    header, records = parse_file(args.path)
    print(f"file: {args.path} ({args.path.stat().st_size} bytes)")
    print(
        f"header: version={header.format_version} n={header.n} m={header.m} "
        f"nnz_jac={header.nnz_jac} nnz_h={header.nnz_h} "
        f"name={header.name!r}"
    )
    print(f"records: {len(records)}")
    for r in records:
        print(
            f"  iter={r.iter:3d} mu={r.mu:.3e} f={r.f:.6e} "
            f"inf_pr={r.inf_pr:.2e} inf_du={r.inf_du:.2e} "
            f"alpha_pr={r.alpha_pr:.2e} alpha_du={r.alpha_du:.2e} "
            f"|x|={len(r.x)} |s|={len(r.s)} |y_c|={len(r.y_c)} |y_d|={len(r.y_d)} "
            f"|z_L|={len(r.z_L)} |z_U|={len(r.z_U)} "
            f"|v_L|={len(r.v_L)} |v_U|={len(r.v_U)}"
        )
        if args.full:
            print(f"    x   = {r.x}")
            print(f"    y_c = {r.y_c}")
            print(f"    y_d = {r.y_d}")
            print(f"    z_L = {r.z_L}")
            print(f"    z_U = {r.z_U}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
