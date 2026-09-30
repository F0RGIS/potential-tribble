# Architecture plugins

Tribble detects about 40 super-resolution architectures automatically through
[spandrel](https://github.com/chaiNNer-org/spandrel). Examples include ESRGAN, Real-ESRGAN,
SwinIR, HAT, DAT, OmniSR, SPAN, Compact/SRVGG, RealCUGAN and more. Plugins cover everything
else, for example your own research model.

Plugins are loaded from:

- `./plugins` (this folder)
- `~/.tribble/plugins`
- any folder listed in the `TRIBBLE_PLUGINS` environment variable

## Writing a plugin

A plugin is a `.py` file with a `build()` function and, optionally, `NAME` and
`detect()`:

```python
NAME = "my-arch"                      # defaults to the file name

def detect(state_dict) -> bool:       # optional: recognise the weights
    return "my_arch.head.weight" in state_dict

def build(state_dict, options):
    model = MyArch(**options)         # options come from the sidecar JSON
    model.load_state_dict(state_dict)
    return model                      # or a dict, see below
```

`build` may return a dict to describe the model instead of letting Tribble probe it:

| key             | meaning                                                   |
|-----------------|-----------------------------------------------------------|
| `model`         | the `nn.Module` (required)                                |
| `scale`         | upscale factor; probed with a test tensor if missing      |
| `in_channels`   | 1 (luma), 3 (RGB) or 4 (RGBA); frames are adapted to fit  |
| `out_channels`  | as above                                                  |
| `pad_multiple`  | input sides must be a multiple of this                    |
| `supports_half` | `False` to always run in fp32                             |

The model gets `float` tensors shaped `(B, C, H, W)` with values in `[0, 1]` and RGB channel
order (set `"channel_order": "bgr"` in the sidecar to change that). It should return the same
layout.

## Choosing a plugin

1. **Auto-detection.** If spandrel doesn't recognise the weights, each plugin's `detect()` is
   tried in turn.
2. **Explicit choice.** Put a sidecar file next to the weights, named `model.pth.json` or
   `model.json`:

```json
{
  "name": "My fancy model",
  "arch": "my-arch",
  "options": {"num_blocks": 12, "scale": 4}
}
```

See [`example_srcnn.py`](example_srcnn.py) for a complete, working example.
