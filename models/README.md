# Models folder

Put model files here and they'll appear in the GUI's model list and in `tribble models`.
Subfolders are scanned too.

Supported formats:

| extension                                  | loaded via                                                                           |
|--------------------------------------------|--------------------------------------------------------------------------------------|
| `.pth` `.pt` `.ckpt` `.bin` `.safetensors` | spandrel auto-detection, then [arch plugins](../plugins/README.md)                   |
| `.pt` `.jit` `.torchscript`                | TorchScript                                                                          |
| `.pt2`                                     | `torch.export` archives                                                              |
| `.onnx`                                    | ONNX Runtime (CUDA, TensorRT, DirectML, CoreML or CPU); fixed-size models are tiled automatically |
| `.pth` (full pickle)                       | only with "Allow fully pickled models" / `--unsafe-pickle`                           |
| `flownet*.pkl` / `rife*.pth`               | RIFE v4.2 to v4.26 frame interpolation (`tribble download rife-v4.26`)               |

Where to get models:

- `tribble download` fetches a few Real-ESRGAN upscalers and RIFE interpolation models.
- [OpenModelDB](https://openmodeldb.info) has hundreds of community models for anime, film,
  compression artifacts, denoising and more.

Optional **sidecar JSON** (`model.pth.json` or `model.json`) overrides detection:

```json
{
  "name": "Display name",
  "scale": 4,
  "half": false,
  "channel_order": "bgr",
  "pad_multiple": 16,
  "tile_size": 256,
  "fixed_size": [256, 256],
  "arch": "plugin-name",
  "options": {},
  "kind": "interpolation"
}
```

A model's kind (`upscale` or `interpolation`) decides which list it appears in. The kind is
guessed from the file name: names containing `rife`, `flownet`, `ifnet` or `interp` count as
interpolation. Set `"kind"` in the sidecar to override the guess.

Other folders can be added with the `TRIBBLE_MODELS` environment variable (separate paths with
`:` or `;` depending on your OS), with **Models → Add model folder…** in the GUI, or with
`tribble models --dir`. Downloads go to `~/.tribble/models`.
