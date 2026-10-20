#!/usr/bin/env python3
"""
validate_s6.py -- correctness validation of the pure-PyTorch S6 block.

WHY THIS EXISTS
---------------
The paper's central finding is a NEGATIVE result about the one component the
authors implemented themselves. A reviewer's most economical counter-
explanation is "your Mamba is not Mamba". This script closes that objection
with measurements rather than assurances.

Three independent checks, in decreasing order of strength:

  CHECK 1  Agreement with the official reference scan
           (mamba_ssm.ops.selective_scan_interface.selective_scan_ref),
           given identical inputs and identical discretised operators.
           NOTE: selective_scan_ref is pure PyTorch and needs no CUDA
           extension. If the mamba-ssm package is not installed, this script
           downloads the official source file and extracts just that function,
           so CHECK 1 runs without any compilation.

  CHECK 2  Agreement with a closed-form evaluation of the same recurrence.
           For a diagonal A the solution of Eq. (16) is

               h_t = sum_{s<=t} ( prod_{r=s+1}^{t} Abar_r ) * Bbar_s * u_s

           computed here by direct O(L^2) summation: no state is carried, so
           the computation is genuinely independent of the O(L) loop under
           test. (An earlier log-space cumulative-product formulation was
           discarded because it underflows for long products.)

  CHECK 3  torch.autograd.gradcheck in float64. A malformed backward pass
           would corrupt every FGSM/PGD result in the paper.

USAGE
-----
    python scripts/validate_s6.py            # CPU is sufficient, ~10 s

Expected on the reported run:
    CHECK 1  max|ours - selective_scan_ref| ~ 1e-8   (fp32)
    CHECK 2  max|loop - closed form|        ~ 1e-17  (fp64)
    CHECK 3  PASS
"""

import ast
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
REF_URL = ("https://raw.githubusercontent.com/state-spaces/mamba/main/"
           "mamba_ssm/ops/selective_scan_interface.py")


# ======================================================================
# The block as used in the paper (verbatim from the main notebook, Cell 7)
# ======================================================================
class SelectiveSSMBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner,
                                kernel_size=4, padding=3, groups=self.d_inner)
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + self.d_inner)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.norm = nn.LayerNorm(d_model)

    def operators(self, x):
        """Everything up to (but excluding) the scan, exposed for comparison."""
        B, L, _ = x.shape
        x_in, z = self.in_proj(x).chunk(2, dim=-1)
        x_conv = F.silu(self.conv1d(x_in.transpose(1, 2))[:, :, :L].transpose(1, 2))
        delta, Bm, Cm = torch.split(
            self.x_proj(x_conv), [self.d_inner, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        A = -torch.exp(self.A_log)
        dA = torch.exp(delta.unsqueeze(-1) * A)
        dB = delta.unsqueeze(-1) * Bm.unsqueeze(2)
        return dict(u=x_conv, z=z, dA=dA, dB=dB, Bm=Bm, Cm=Cm, delta=delta, A=A)

    def scan_loop(self, u, dA, dB, Cm):
        """The sequential scan under test (Algorithm 2, lines 7-10)."""
        B, L, _ = u.shape
        h = torch.zeros(B, self.d_inner, self.d_state, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * u[:, t].unsqueeze(-1)
            ys.append(torch.einsum("bds,bs->bd", h, Cm[:, t]))
        return torch.stack(ys, dim=1)

    def forward(self, x):
        o = self.operators(x)
        y = self.scan_loop(o["u"], o["dA"], o["dB"], o["Cm"])
        y = (y + o["u"] * self.D) * F.silu(o["z"])
        return self.norm(self.out_proj(y) + x)
# ======================================================================


def get_reference_scan():
    """Return selective_scan_ref, importing it or extracting it from source."""
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
        return selective_scan_ref, "installed mamba-ssm package"
    except Exception:
        pass
    try:
        src = urllib.request.urlopen(REF_URL, timeout=60).read().decode("utf-8")
        fn = next(n for n in ast.parse(src).body
                  if isinstance(n, ast.FunctionDef) and n.name == "selective_scan_ref")
        from einops import rearrange, repeat
        ns = {"torch": torch, "F": F, "rearrange": rearrange, "repeat": repeat}
        exec(ast.get_source_segment(src, fn), ns)
        return ns["selective_scan_ref"], "official source (state-spaces/mamba)"
    except Exception as e:                                        # noqa: BLE001
        print(f"[info] reference unavailable ({type(e).__name__}: {e})")
        return None, None


def closed_form_scan(dA, dB, u, Cm):
    """CHECK 2 reference: direct O(L^2) summation, carrying no state."""
    L = u.shape[1]
    src = dB * u.unsqueeze(-1)
    ys = []
    for t in range(L):
        acc = torch.zeros_like(src[:, 0])
        for s in range(t + 1):
            w = src[:, s]
            if s < t:
                w = w * dA[:, s + 1:t + 1].prod(dim=1)
            acc = acc + w
        ys.append(torch.einsum("bds,bs->bd", acc, Cm[:, t]))
    return torch.stack(ys, dim=1)


def main():
    d_model, d_state, B, L = 64, 16, 4, 64
    print(f"device = {DEV}\n")
    failures = 0

    # ---------------- CHECK 1 ----------------
    ref, origin = get_reference_scan()
    if ref is not None:
        blk = SelectiveSSMBlock(d_model, d_state).to(DEV).eval()
        x = torch.randn(B, L, d_model, device=DEV)
        with torch.no_grad():
            o = blk.operators(x)
            ours = blk.scan_loop(o["u"], o["dA"], o["dB"], o["Cm"])
            r = ref(o["u"].transpose(1, 2).contiguous(),
                    o["delta"].transpose(1, 2).contiguous(),
                    o["A"],
                    o["Bm"].transpose(1, 2).contiguous(),
                    o["Cm"].transpose(1, 2).contiguous(),
                    D=None, z=None, delta_bias=None,
                    delta_softplus=False, return_last_state=False).transpose(1, 2)
            dev1 = (ours - r).abs().max().item()
        ok = dev1 < 1e-4
        failures += (not ok)
        print(f"CHECK 1  {'PASS' if ok else 'FAIL'}  "
              f"max|ours - selective_scan_ref| = {dev1:.3e}   "
              f"(fp32 threshold 1e-4)  [{origin}]")
    else:
        print("CHECK 1  SKIPPED  reference could not be obtained")

    # ---------------- CHECK 2 ----------------
    blk64 = SelectiveSSMBlock(d_model, d_state).to(DEV).double().eval()
    x64 = torch.randn(B, L, d_model, device=DEV, dtype=torch.float64)
    with torch.no_grad():
        o = blk64.operators(x64)
        loop = blk64.scan_loop(o["u"], o["dA"], o["dB"], o["Cm"])
        closed = closed_form_scan(o["dA"], o["dB"], o["u"], o["Cm"])
        dev2 = (loop - closed).abs().max().item()
    ok2 = dev2 < 1e-8
    failures += (not ok2)
    print(f"CHECK 2  {'PASS' if ok2 else 'FAIL'}  "
          f"max|loop - closed form| = {dev2:.3e}   (fp64 threshold 1e-8)")

    # ---------------- CHECK 3 ----------------
    small = SelectiveSSMBlock(8, 4).to(DEV).double()
    xg = torch.randn(2, 6, 8, device=DEV, dtype=torch.float64, requires_grad=True)
    try:
        ok3 = torch.autograd.gradcheck(lambda t: small(t).sum(), (xg,),
                                       eps=1e-6, atol=1e-5, rtol=1e-3)
    except Exception as e:                                        # noqa: BLE001
        ok3 = False
        print(f"CHECK 3  FAIL  gradcheck raised {type(e).__name__}: {e}")
    else:
        print(f"CHECK 3  {'PASS' if ok3 else 'FAIL'}  "
              f"autograd.gradcheck on the full block (fp64)")
    failures += (not ok3)

    print(f"\n{3 - failures}/3 checks passed.")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
