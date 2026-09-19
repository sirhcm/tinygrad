# RX 9060 HDMI animation

Render a shader at 1280×720, 60 Hz on the USB3-attached RX 9060 (`1002:7590`).
The monitor connects directly to the GPU. FTDI is not required for normal operation.

```sh
python -m extra.amd_display animate --shader fractal --frames 600
python -m extra.amd_display animate --shader refraction --frames 6000
python -m extra.amd_display animate --shader refraction --frames 0
```

`--frames 0` runs indefinitely; Ctrl-C or SIGTERM requests a clean stop. A positive count limits the run;
600 frames is ten seconds when rendering meets the frame budget. `--device N` selects the USB GPU.
DMCUB firmware is downloaded and hash-verified; `--firmware PATH` uses a local copy of the same image.
Output is fixed at 720p60; there is no EDID negotiation.

## Rendering and scanout

Python initializes the GPU, display firmware, HDMI link and two framebuffers in one VRAM allocation.
Each framebuffer is a tinygrad `Buffer` of 1280×720 `uint32` pixels: XRGB8888 (`0x00RRGGBB`,
little-endian). Compute and display access the same allocation; pixels never pass over USB.

While scanout reads one buffer, the GPU renders into the other. It flushes the completed image,
programs its address for the next vblank, and waits until that address is active before reusing
the old buffer. Rendering does not need to fit inside vblank, and scanout never reads a buffer
while the render kernel modifies it.

The animation compiles both buffer bindings into an HCQ2 conditional loop. Each frame renders,
flips, advances the counter and checks the stop word. The command processor repeats while that word is zero. A finite
run sets that word on its last iteration; the host can also set it at any time. Both exits reach the
normal HCQ2 completion signal before buffers or display hardware are released.

`display_queue(dev)` installs the GPU branch and register-wait encoder. The host submits once and
watches for completion or cancellation at 4 Hz; it never drives individual frames. Command storage
and application memory stay constant for any duration. Flip commands live in `extra/amd_display`;
there is no `PRESENT` operation or timing collection. Stop is cooperative: a stalled kernel
or display wait can prevent the queue from reaching its stop check.

The application uses one uncached, contiguous VRAM allocation, `display.memory`, with typed views:

| Byte offset | Contents | View |
| --- | --- | --- |
| 0 | `uint32` stop: write 1 to exit; bytes 4–7 reserved | `display.stop` |
| 8 | `uint64` completed frame count | `display.counter` |
| 16–4095 | Reserved for input and persistent state | |
| 4096 | XRGB8888 framebuffer A | `display.buffers[0]` |
| 3690496 | XRGB8888 framebuffer B | `display.buffers[1]` |

The stop write must use direct memory access (`display.stop.host.view(fmt="I")[0] = 1`), followed by
`display.adev.gmc.flush_hdp()`. A queued copy behind the running graph cannot stop it. GPU-owned
state and pixels occupy separate regions so future SRAM input snapshots can preserve them.

Shaders live in `shaders/NAME.py` and implement `shade(frag_coord, resolution, time)`, returning three
RGB UOps. `--shader NAME` imports the module directly. Time is the 64-bit GPU frame counter divided by 60;
coordinates use pixel centers and a bottom-left origin. RGB is clamped and packed by `render_kernel`.
Shader time is still float32, so its precision decreases over multi-day runs.

## Files

- `__main__.py`: device lifetime, CLI and execution.
- `bios.py`: VBIOS HDMI routing and reference clock.
- `dmub.py`: register access, display firmware and command ring.
- `hdmi.py`: HDMI clock, timing and encoder setup.
- `scanout.py`: framebuffer allocation, display memory-fetch pipeline and flip commands.
- `queue.py`: GPU command loops, cache writeback and register waits.
- `animation.py`: render graph, refresh pacing, frame counter and stop condition.
- `shaders/`: shaders and framebuffer kernel wrapper.
- `autogen/`: generated register definitions and their generator.

Regenerate registers from Linux v6.16:

```sh
python extra/amd_display/autogen/generate.py /path/to/linux-v6.16
```

Shutdown stops scanout, HDMI, DMCUB and the compute/SDMA queues, and leaves the GPU initialized for
the next run. It calls `AMDev.fini(lower_clocks=False)` and retains the display clock and power
configuration; low-power shutdown is not implemented. USB stays open through Python's final buffer cleanup.

If the GPU stalls during development, power-cycle the eGPU or use the optional FTDI recovery command
after exiting the process:

```sh
python extra/usbgpu/debug.py --reset --wait --no-read
```
