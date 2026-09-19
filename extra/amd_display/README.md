# RX 9060 HDMI animation

Render a shader at 1280×720, 60 Hz on the USB3-attached RX 9060 (`1002:7590`).
The monitor connects directly to the GPU. A provisioned FTDI adapter resets the board on exit.

```sh
python -m extra.amd_display animate --shader fractal --frames 600
python -m extra.amd_display animate --shader refraction --frames 6000
```

`--frames` must be positive and even; 600 frames is ten seconds. `--device N` selects the USB GPU.
`--ftdi URL` selects the reset adapter. DMCUB firmware is downloaded and hash-verified;
`--firmware PATH` uses a local copy of the same image. Output is fixed at 720p60; there is no EDID negotiation.

## Rendering and presentation

Python initializes the GPU, display firmware, HDMI link and two contiguous VRAM framebuffers.
Each framebuffer is a tinygrad `Buffer` of 1280×720 `uint32` pixels: XRGB8888 (`0x00RRGGBB`,
little-endian). Compute and display access the same allocation; pixels never pass over USB.

The animation compiles a two-frame body—render A, present A, advance; render B, present B, advance—
inside an HCQ2 `RANGE`. The display backend encodes a GPU branch, so command storage stays constant
as the frame count increases. The host submits once and waits for completion.

```python
fb = UOp.from_buffer(display.buffers[0])
present = UOp.custom_function("present", UOp.const(0)).call(fb)
```

`presentation(display)` installs the backend during compilation and execution. Presentation flushes
render writes, updates the scanout address and waits for vblank and flip completion on the compute queue.
Scanout owns the presented buffer until the next presentation; render into the other buffer.

Shaders live in `shaders/NAME.py` and implement `shade(frag_coord, resolution, time)`, returning three
RGB UOps. `--shader NAME` imports the module directly. Time is the GPU frame counter divided by 60;
coordinates use pixel centers and a bottom-left origin. RGB is clamped and packed by `render_kernel`.

After completion, GPU timestamps report mean/p95/max render time, the fraction of the 16.67 ms budget
used, and estimated missed refreshes. Rendering time excludes presentation waits and cache writeback.
Timing history uses 48 bytes per frame and is read back once.

## Files

- `__main__.py`: device lifetime, CLI and execution.
- `bios.py`: VBIOS HDMI routing and reference clock.
- `dmub.py`: register access, display firmware and command ring.
- `hdmi.py`: HDMI clock, timing and encoder setup.
- `scanout.py`: framebuffer allocation and display memory-fetch pipeline.
- `present.py`: HCQ2 presentation and GPU command loops.
- `animation.py`: render/present graph, frame counter and timing archive.
- `shaders/`: shaders and framebuffer kernel wrapper.
- `autogen/`: generated register definitions and their generator.

Regenerate registers from Linux v6.16:

```sh
python extra/amd_display/autogen/generate.py /path/to/linux-v6.16
```

Normal `AMDev.fini()` clock reduction stalls after display use, so shutdown stops scanout and HDMI,
closes USB and resets through FTDI. If a run is interrupted before cleanup, exit the process and reset:

```sh
python extra/usbgpu/debug.py --reset --wait --no-read
```
