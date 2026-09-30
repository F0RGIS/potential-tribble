"""Command line interface.

Examples::

    tribble upscale in.mp4 -m models/realesr-animevideov3.pth
    tribble upscale in.mp4 -m a.pth -m b.onnx --strength 1 0.6 --tile 512 -o out.mkv
    tribble upscale *.mp4 --preset presets/anime-fast.json --out-dir upscaled/
    tribble models
    tribble info models/4x_foo.safetensors
    tribble download realesr-animevideov3
    tribble gui
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from pathlib import Path
from typing import List, Optional

from . import __version__
from .config import JobConfig, ModelStep


def _add_upscale_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("inputs", nargs="+", help="input video(s) or image(s)")
    p.add_argument("-o", "--output", help="output file (single input only)")
    p.add_argument("--out-dir", help="output folder for batch runs")
    p.add_argument("--preset", help="JSON preset to start from (flags override it)")
    p.add_argument("--save-preset", help="write the effective settings to this JSON file and continue")

    g = p.add_argument_group("models")
    g.add_argument("-m", "--model", action="append", default=[],
                   help="model file or builtin:<mode>@<scale>; repeat to chain models")
    g.add_argument("--strength", type=float, nargs="+",
                   help="per-model blend strength 0..1 (one per -m)")

    g = p.add_argument_group("processing")
    g.add_argument("--device", help="auto, cpu, cuda, cuda:1, mps")
    g.add_argument("--fp32", action="store_true", help="disable half precision")
    g.add_argument("--tile", type=int, help="tile size in pixels (0 = whole frame)")
    g.add_argument("--tile-overlap", type=int)
    g.add_argument("--bit-depth", type=int, choices=[8, 16], help="precision of frame pipes")
    g.add_argument("--unsafe-pickle", action="store_true", help="allow loading fully pickled models")

    g = p.add_argument_group("input")
    g.add_argument("--start", type=float, help="start time in seconds")
    g.add_argument("--end", type=float, help="end time in seconds")
    g.add_argument("--frames", type=int, help="process at most N frames")
    g.add_argument("--pre-scale", type=float, help="downscale factor before the model, e.g. 0.5")
    g.add_argument("--pre-filters", help='ffmpeg filters on decode, e.g. "yadif,hqdn3d=2"')

    g = p.add_argument_group("output")
    g.add_argument("--scale", type=float, help="final scale relative to the source")
    g.add_argument("--width", type=int, help="final width (keeps aspect)")
    g.add_argument("--height", type=int, help="final height (keeps aspect)")
    g.add_argument("--resize-filter", help="ffmpeg scale flags for final resize (lanczos, spline, ...)")
    g.add_argument("--post-filters", help='ffmpeg filters before encoding, e.g. "unsharp=5:5:0.5"')
    g.add_argument("--container", help="mkv, mp4, mov, webm")
    g.add_argument("--codec", help="ffmpeg video encoder, e.g. libx265, hevc_nvenc")
    g.add_argument("--crf", type=int)
    g.add_argument("--bitrate", help="target bitrate instead of CRF, e.g. 25M")
    g.add_argument("--enc-preset", help="encoder preset, e.g. slow, p7, 6")
    g.add_argument("--pix-fmt", help="output pixel format, e.g. yuv420p10le")
    g.add_argument("--fps", type=float, help="override output frame rate")
    g.add_argument("--no-audio", action="store_true")
    g.add_argument("--audio-codec", help="copy (default), aac, libopus, flac ...")
    g.add_argument("--no-subs", action="store_true")
    g.add_argument("--extra-args", help="raw extra ffmpeg encoder arguments")
    g.add_argument("--png-sequence", action="store_true", help="write PNG frames instead of a video")
    g.add_argument("-y", "--overwrite", action="store_true")


def build_config(a: argparse.Namespace) -> JobConfig:
    cfg = JobConfig.load(a.preset) if a.preset else JobConfig()
    if a.model:
        strengths = a.strength or []
        cfg.models = [
            ModelStep(m, strengths[i] if i < len(strengths) else 1.0) for i, m in enumerate(a.model)
        ]
    elif a.strength:
        for step, s in zip(cfg.models, a.strength):
            step.strength = s

    p, i, o = cfg.processing, cfg.input, cfg.output
    _set(p, "device", a.device)
    if a.fp32:
        p.half_precision = False
    _set(p, "tile_size", a.tile)
    _set(p, "tile_overlap", a.tile_overlap)
    _set(p, "bit_depth", a.bit_depth)
    if a.unsafe_pickle:
        p.allow_unsafe_pickle = True

    _set(i, "start_time", a.start)
    _set(i, "end_time", a.end)
    _set(i, "max_frames", a.frames)
    _set(i, "pre_scale", a.pre_scale)
    _set(i, "pre_filters", a.pre_filters)

    if a.scale:
        o.size_mode, o.scale = "scale", a.scale
    elif a.width and a.height:
        o.size_mode, o.width, o.height = "exact", a.width, a.height
    elif a.width:
        o.size_mode, o.width = "width", a.width
    elif a.height:
        o.size_mode, o.height = "height", a.height
    _set(o, "resize_filter", a.resize_filter)
    _set(o, "post_filters", a.post_filters)
    _set(o, "container", a.container)
    _set(o, "codec", a.codec)
    _set(o, "crf", a.crf)
    _set(o, "bitrate", a.bitrate)
    _set(o, "preset", a.enc_preset)
    _set(o, "pixel_format", a.pix_fmt)
    _set(o, "fps", a.fps)
    _set(o, "audio_codec", a.audio_codec)
    _set(o, "extra_args", a.extra_args)
    if a.no_audio:
        o.copy_audio = False
    if a.no_subs:
        o.copy_subtitles = False
    if a.png_sequence:
        o.image_sequence = True
    if a.overwrite:
        o.overwrite = True
    return cfg


def _set(obj, attr: str, value) -> None:
    if value is not None:
        setattr(obj, attr, value)


def cmd_upscale(a: argparse.Namespace) -> int:
    from .inference import Upscaler
    from .pipeline import Cancelled, default_output_path, run_job

    cfg = build_config(a)
    if not cfg.models:
        print("error: no model given (-m or a preset with models)", file=sys.stderr)
        return 2
    if a.output and len(a.inputs) > 1:
        print("error: -o only works with a single input; use --out-dir", file=sys.stderr)
        return 2
    if a.save_preset:
        cfg.save(a.save_preset)
        print(f"Saved preset to {a.save_preset}")

    upscaler = Upscaler.from_config(cfg, print)
    cancel = threading.Event()
    failures = 0
    for inp in a.inputs:
        out = Path(a.output) if a.output else default_output_path(inp, cfg, a.out_dir)
        try:
            run_job(inp, out, cfg, upscaler, progress=_ProgressBar(), log_fn=print, cancel=cancel)
        except (KeyboardInterrupt, Cancelled):
            cancel.set()
            print("\nCancelled.", file=sys.stderr)
            return 130
        except Exception as exc:
            failures += 1
            print(f"\nerror: {inp}: {exc}", file=sys.stderr)
    return 1 if failures else 0


class _ProgressBar:
    def __call__(self, pr) -> None:
        width = 30
        filled = int(width * pr.fraction)
        eta = f"{int(pr.eta // 60):d}:{int(pr.eta % 60):02d}"
        sys.stderr.write(
            f"\r[{'#' * filled}{'.' * (width - filled)}] {pr.frame}/{pr.total} "
            f"{pr.fps:5.2f} fps  ETA {eta}   "
        )
        if pr.frame >= pr.total:
            sys.stderr.write("\n")
        sys.stderr.flush()


def cmd_models(a: argparse.Namespace) -> int:
    from .models import BUILTIN_MODELS, model_dirs, scan_models
    from .models.plugins import load_plugins
    from .models.registry import plugin_dirs

    dirs = model_dirs(a.dir)
    print("Model folders:")
    for d in dirs:
        print(f"  {'*' if d.is_dir() else ' '} {d}")
    entries = scan_models(dirs)
    print(f"\nModels ({len(entries)}):")
    for e in entries:
        print(f"  {e.name:<40} {e.size_bytes / 1e6:8.1f} MB  {e.path}")
    print("\nBuiltins:")
    for b in BUILTIN_MODELS:
        print(f"  {b}")
    plugins = load_plugins(plugin_dirs())
    print(f"\nArch plugins ({len(plugins)}):")
    for name, pl in plugins.items():
        print(f"  {name:<20} {pl.path}")
    return 0


def cmd_info(a: argparse.Namespace) -> int:
    from .models import load_model

    m = load_model(a.model, device=a.device or "auto", half=False, allow_unsafe_pickle=a.unsafe_pickle)
    print(m.describe())
    print(f"  scale:        x{m.scale:g}")
    print(f"  channels:     {m.in_channels} -> {m.out_channels}")
    print(f"  pad multiple: {m.pad_multiple}")
    if m.fixed_size:
        print(f"  fixed input:  {m.fixed_size[1]}x{m.fixed_size[0]}")
    for k, v in m.info.items():
        print(f"  {k}: {v}")
    return 0


def cmd_probe(a: argparse.Namespace) -> int:
    from .video import probe

    for p in a.inputs:
        print(f"{p}: {probe(p).summary()}")
    return 0


def cmd_download(a: argparse.Namespace) -> int:
    from .catalog import CATALOG, download

    if not a.names:
        for m in CATALOG.values():
            print(f"  {m.key:<28} x{m.scale}  {m.description}")
        return 0
    for name in a.names:
        def prog(got, total):
            pct = f"{100 * got / total:5.1f}%" if total else f"{got / 1e6:.1f} MB"
            sys.stderr.write(f"\r{name}: {pct}")
        path = download(name, Path(a.dir) if a.dir else None, prog)
        sys.stderr.write("\n")
        print(f"Saved {path}")
    return 0


def cmd_gui(a: argparse.Namespace) -> int:
    from .gui.app import main as gui_main

    return gui_main([])


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="tribble", description="Local AI video upscaler")
    ap.add_argument("--version", action="version", version=f"tribble {__version__}")
    ap.add_argument("-v", "--verbose", action="store_true", help="debug logging (shows ffmpeg commands)")
    sub = ap.add_subparsers(dest="command")

    p = sub.add_parser("upscale", help="upscale videos or images")
    _add_upscale_args(p)
    p.set_defaults(func=cmd_upscale)

    p = sub.add_parser("models", help="list available models and plugins")
    p.add_argument("--dir", action="append", help="extra model folder")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("info", help="load a model and show what was detected")
    p.add_argument("model")
    p.add_argument("--device")
    p.add_argument("--unsafe-pickle", action="store_true")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("probe", help="show media info")
    p.add_argument("inputs", nargs="+")
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("download", help="download a model from the built-in catalog")
    p.add_argument("names", nargs="*", help="catalog keys (omit to list)")
    p.add_argument("--dir", help="destination folder (default ~/.tribble/models)")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("gui", help="launch the graphical interface")
    p.set_defaults(func=cmd_gui)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    ap = make_parser()
    a = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    if not getattr(a, "func", None):
        ap.print_help()
        return 0
    try:
        return a.func(a)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        if a.verbose:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
