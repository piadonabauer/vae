#!/usr/bin/env python3
"""Latent statistics as a sampleability proxy (eval-only, final wave @300).

For each trained arm (and the pretrained zero-shot floor), encode the 10 fixed
val clips and report distribution health of the posterior:
  - per-dim KL to N(0,1): 0.5*(mu^2 + sigma^2 - 1 - 2*log sigma)
  - channel-wise std of mu (how spread the code is) and mean sigma
  - active channels: fraction of channels with std(mu) > 0.1 (KL-collapse proxy)
A latent that a generative prior could sample from should have mu-std near 1
and sigma not collapsed to ~0 or blown past 1.

Weights: ema.pt from the epoch-299 checkpoint (final eval uses EMA), falling
back to the sharded `model/` dir if ema.pt cannot be read as a state dict.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

OS_ROOT = Path("/home/coder/vae/Open-Sora")
sys.path.insert(0, str(OS_ROOT))

from opensora.registry import MODELS, build_module  # noqa: E402
from opensora.utils.ckpt import load_checkpoint  # noqa: E402
import opensora.models.vae.wan_video_vae  # noqa: F401,E402  (registers the model)

OUT_DIR = Path(__file__).resolve().parent
OUTPUTS = OS_ROOT / "outputs"

# name -> (jobdir, ckpt subdir or None for pretrained-only)
ARMS = {
    "pretrained zero-shot (per-view, TC on)": ("paper_E1z_perview_zeroshot_tcON__job1791026450_t50", None),
    "per-view 16-ch (@300)": ("paper_E1b_perview_tcT__job1791249364_t2", "epoch299-global_step1912"),
    "per-view 32-ch ctrl (@300)": ("paper_Ectrl_perview_tcT_widen32__job1791271134_t53", "epoch299-global_step1912"),
    "per-view 64-ch ctrl (@300)": ("paper_Ectrl_perview_tcT_widen64__job1791298425_t54", "epoch299-global_step1912"),
    "fused 16-ch (@300)": ("paper_E1d_fused_tcT__job1791225734_t4", "epoch299-global_step1950"),
    "fused 32-ch (@300)": ("paper_E11a_fused_tcT_widen32__job1791237549_t8", "epoch299-global_step1950"),
    "fused 64-ch (@300)": ("paper_E11b_fused_tcT_widen64__job1791189563_t9", "epoch299-global_step1950"),
    "fused 32-ch + diff-loss (@300)": ("paper_E_combo_diffLoss_widen32__job1791201374_t36", "epoch299-global_step1950"),
    "fused all tweaks (@300)": ("paper_Ebest_allcombined__job1791213193_t52", "epoch299-global_step1950"),
}

GT_DUMP = OUTPUTS / "paper_E1d_fused_tcT__job1791225734_t4" / "final_eval_dump_val.pt"


def load_val_clips() -> torch.Tensor:
    d = torch.load(GT_DUMP, map_location="cpu")
    gts = [c["gt"] for c in d["clips"]]  # each [V, C, T, H, W] uint8
    x = torch.stack(gts).float() / 255.0 * 2.0 - 1.0  # [N, V, C, T, H, W] in [-1,1]
    return x


def load_weights(model, jobdir: Path, ckpt_sub: str):
    ckpt_dir = jobdir / ckpt_sub
    ema_path = ckpt_dir / "ema.pt"
    if ema_path.exists():
        try:
            sd = torch.load(ema_path, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            missing, unexpected = model.load_state_dict(sd, strict=False)
            n_model = len(dict(model.state_dict()))
            if len(missing) < 0.5 * n_model:
                print(f"  loaded EMA ({len(missing)} missing / {len(unexpected)} unexpected)")
                return "ema"
            print(f"  ema.pt looked wrong ({len(missing)}/{n_model} missing) -> sharded fallback")
        except Exception as e:  # noqa: BLE001
            print(f"  ema.pt load failed ({e}) -> sharded fallback")
    load_checkpoint(model, str(ckpt_dir / "model"), device_map="cpu")
    return "model"


@torch.no_grad()
def stats_for(model, x: torch.Tensor, device) -> dict:
    mus, sigmas = [], []
    for i in range(x.shape[0]):
        xi = x[i : i + 1].to(device, torch.bfloat16)
        out = model(xi)
        posterior = out[1]
        if isinstance(posterior, (tuple, list)) and len(posterior) == 2:
            mu, logvar = posterior
        else:
            mu, logvar = posterior.mean, posterior.logvar
        mus.append(mu.float().cpu())
        sigmas.append(torch.exp(0.5 * logvar.float()).cpu())
    mu = torch.cat(mus)      # [N, Cz, T', H', W'] (fused) or with a view dim folded in
    sigma = torch.cat(sigmas)
    # channel axis = dim 1 regardless of layout
    red = [d for d in range(mu.dim()) if d != 1]
    mu_std_per_ch = mu.std(dim=red)
    sigma_mean_per_ch = sigma.mean(dim=red)
    kl = 0.5 * (mu.pow(2) + sigma.pow(2) - 1.0 - 2.0 * sigma.clamp_min(1e-8).log())
    # Per-channel SNR: a low-amplitude channel still carries information if its
    # posterior sigma is even smaller. SNR >> 1 = informative; ~<=1 = noise-level.
    snr_per_ch = mu_std_per_ch / sigma_mean_per_ch.clamp_min(1e-8)
    return {
        "snr_active_frac": float((snr_per_ch > 2.0).float().mean()),
        "snr_per_ch": [round(float(v), 2) for v in snr_per_ch],
        "sigma_mean_per_ch": [round(float(v), 4) for v in sigma_mean_per_ch],
        "latent_shape": list(mu.shape[1:]),
        "n_channels": int(mu.shape[1]),
        "mu_mean": float(mu.mean()),
        "mu_std_overall": float(mu.std()),
        "mu_std_per_ch_min": float(mu_std_per_ch.min()),
        "mu_std_per_ch_med": float(mu_std_per_ch.median()),
        "mu_std_per_ch_max": float(mu_std_per_ch.max()),
        "sigma_mean": float(sigma.mean()),
        "sigma_mean_per_ch_min": float(sigma_mean_per_ch.min()),
        "sigma_mean_per_ch_max": float(sigma_mean_per_ch.max()),
        "kl_per_dim": float(kl.mean()),
        "active_channels_frac": float((mu_std_per_ch > 0.1).float().mean()),
        "mu_std_per_ch": [round(float(v), 4) for v in mu_std_per_ch],
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = load_val_clips()
    print("val clips:", tuple(x.shape))
    results = {}
    for name, (jobdir, ckpt_sub) in ARMS.items():
        jd = OUTPUTS / jobdir
        cfg = json.loads((jd / "config.txt").read_text())["model"]
        print(f"== {name} ==")
        model = build_module(cfg, MODELS, device_map="cpu", torch_dtype=torch.bfloat16)
        src = "pretrained"
        if ckpt_sub is not None:
            src = load_weights(model, jd, ckpt_sub)
        model = model.to(device, torch.bfloat16).eval()
        r = stats_for(model, x, device)
        r["weights"] = src
        results[name] = r
        print(f"  KL/dim {r['kl_per_dim']:.3f}  mu-std {r['mu_std_overall']:.3f}  "
              f"sigma {r['sigma_mean']:.3f}  active {r['active_channels_frac']*100:.0f}% "
              f"of {r['n_channels']} ch")
        del model
        torch.cuda.empty_cache()

    (OUT_DIR / "latent_stats_finalwave.json").write_text(json.dumps(results, indent=2))
    lines = [
        "# Latent statistics (sampleability proxy), final wave @300",
        "",
        "Encoded the 10 fixed val clips; posterior stats per arm (EMA weights).",
        "KL/dim = mean per-dim KL to N(0,1). Active ch = fraction of channels",
        "with std(mu) > 0.1 (collapse proxy). Healthy-for-sampling: mu-std near 1,",
        "sigma neither ~0 nor >1, no dead channels.",
        "",
        "| arm | ch | KL/dim | std(mu) | mean sigma | active ch | SNR>2 ch | weights |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['n_channels']} | {r['kl_per_dim']:.3f} | "
            f"{r['mu_std_overall']:.3f} | {r['sigma_mean']:.3f} | "
            f"{r['active_channels_frac']*100:.0f}% | {r['snr_active_frac']*100:.0f}% | {r['weights']} |"
        )
    (OUT_DIR / "latent_stats_finalwave.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT_DIR / "latent_stats_finalwave.md")


if __name__ == "__main__":
    main()
