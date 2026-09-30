import threading

import pytest

from tribble.cli import main as cli_main
from tribble.config import JobConfig, ModelStep
from tribble.pipeline import Cancelled, run_job
from tribble.video import probe, read_frame_at


def test_probe(sample_video):
    info = probe(sample_video)
    assert (info.width, info.height) == (64, 48)
    assert info.has_audio and abs(info.fps - 10) < 0.01


def test_video_job_with_audio(sample_video, tmp_path):
    cfg = JobConfig(models=[ModelStep("builtin:bicubic@2")])
    out = run_job(sample_video, tmp_path / "out.mkv", cfg, log_fn=lambda s: None)
    info = probe(out)
    assert (info.width, info.height) == (128, 96)
    assert info.has_audio
    assert 9 <= info.frame_count <= 11


def test_16bit_resize_and_range(sample_video, tmp_path):
    cfg = JobConfig(models=[ModelStep("builtin:nearest@2")])
    cfg.processing.bit_depth = 16
    cfg.input.start_time = 0.3
    cfg.input.max_frames = 4
    cfg.output.size_mode = "width"
    cfg.output.width = 100
    cfg.output.pixel_format = "yuv420p10le"
    frames = []
    out = run_job(sample_video, tmp_path / "o.mp4", cfg, progress=frames.append, log_fn=lambda s: None)
    info = probe(out)
    assert (info.width, info.height) == (100, 76)
    assert frames[-1].frame == 4 and "10" in info.pix_fmt


def test_image_in_image_out(sample_video, tmp_path, ffmpeg):
    import subprocess

    img = tmp_path / "a.png"
    subprocess.run([ffmpeg, "-loglevel", "error", "-i", str(sample_video), "-frames:v", "1", str(img)], check=True)
    cfg = JobConfig(models=[ModelStep("builtin:bicubic@4")])
    out = run_job(img, tmp_path / "a_up.png", cfg, log_fn=lambda s: None)
    frame = read_frame_at(str(out), 0)
    assert frame.shape == (192, 256, 3)


def test_existing_output_refused(sample_video, tmp_path):
    out = tmp_path / "exists.mkv"
    out.write_bytes(b"x")
    with pytest.raises(FileExistsError):
        run_job(sample_video, out, JobConfig(models=[ModelStep("builtin:identity")]), log_fn=lambda s: None)


def test_cancel(sample_video, tmp_path):
    cancel = threading.Event()

    def on_progress(p):
        if p.frame >= 2:
            cancel.set()

    cfg = JobConfig(models=[ModelStep("builtin:bicubic@2")])
    cfg.processing.queue_size = 1
    with pytest.raises(Cancelled):
        run_job(sample_video, tmp_path / "c.mkv", cfg, progress=on_progress, cancel=cancel, log_fn=lambda s: None)


def test_cli_roundtrip(sample_video, tmp_path):
    preset = tmp_path / "p.json"
    rc = cli_main([
        "upscale", str(sample_video), "-m", "builtin:bicubic@2", "--frames", "3",
        "--save-preset", str(preset), "-o", str(tmp_path / "cli.mkv"), "--codec", "ffv1", "--pix-fmt", "yuv444p",
    ])
    assert rc == 0
    assert probe(tmp_path / "cli.mkv").codec == "ffv1"
    cfg = JobConfig.load(preset)
    assert cfg.input.max_frames == 3 and cfg.models[0].model == "builtin:bicubic@2"
