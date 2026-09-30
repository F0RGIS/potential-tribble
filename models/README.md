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

Where to get models:

- `tribble download` fetches a few Real-ESRGAN models.
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
  "options": {}
}
```

Other folders can be added with the `TRIBBLE_MODELS` environment variable (separate paths with
`:` or `;` depending on your OS), with **Models → Add model folder…** in the GUI, or with
`tribble models --dir`. Downloads go to `~/.tribble/models`.
