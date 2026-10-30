# gst-plugin-tof

Plugin `qtitof`: two elements for the Arducam ToF camera (B0410, a 240 × 180 indirect time-of-flight module with its own
MCU), read through the upstream Linux camera driver (CAMSS) rather than the camera service.

| Element | What it does |
|---|---|
| `qtitofsrc` | sets up the CAMSS path `msm_csiphyN → msm_csidN → msm_vfeN_rdi0`, streams the raw node and pushes one buffer per group of 4 DCS (phase) frames, `video/x-tof-dcs, format=DCS16`, 30 groups/s; range control while PLAYING |
| `qtitofdepth` | DCS16 groups → calibrated depth (`GRAY16_LE` millimetres or a colour-mapped RGB(A)/GRAY8 view), plus optional `amplitude`, `confidence` and `pointcloud` (QIM SDK tensor) outputs |

Build with `-DENABLE_GST_PLUGIN_TOF=ON` (on by default with `ENABLE_GST_IMSDK_PLUGINS`). It needs only the QIM SDK base
libraries (`gstreamer-qcom-oss-video`, `-allocators`, `-ml`), not the camera service.

## Requirements

- **Kernel:** `qcom-camss` with the CSIPHY, CSID and VFE RDI entities for the connector, and the `arducam-pivariety`
  sensor driver (Raspberry Pi kernel) bound to the module on that CSIPHY. On CAMSS the driver must:
  - offer the frames as `MEDIA_BUS_FMT_SRGGB12_1X12` 240 × 180, because CAMSS has no Y12 format;
  - provide `V4L2_CID_LINK_FREQ`, from which CAMSS derives the CSIPHY settle time;
  - provide the `trigger_mode` control (range).

  The read-only `firmware_version` and `device_id` controls are optional: without them both properties read 0.
- **Permissions:** `/dev/media*`, `/dev/video*` and `/dev/v4l-subdev*` (group `video`), and `/dev/gpiochip*` (group
  `gpiod` or as the distribution sets it) when a CSI mux line is held. No root is needed at run time.
- **CSI mux:** some boards route a connector through a mux GPIO. `mux-gpio=auto` (the default) holds the line of a known
  board for the chosen `csiphy` and touches no GPIO elsewhere. Today the table has one entry: on the Arduino VENTUNO Q
  (`arduino,monza`), TLMM GPIO 24 (`f100000.pinctrl`) = 1 routes CAMERA0 to CSIPHY0. Use `mux-gpio=gpiochipN:LINE=VALUE`
  for another board, or `""` if something else sets the mux.

Tested on an Arduino VENTUNO Q (QCS8300, Ubuntu 24.04, `linux-qcom` 6.8, QIM SDK 0.1.3 and `main`), CAMERA0, module
firmware 0x10002.

## Module behaviour (firmware 0x10002)

- The module sends a burst of 4 frames every 33.3 ms (120 frames/s), whatever `frame-rate` is set to; `exposure` is
  also accepted and ignored. `qtitofdepth temporal-average` is the software substitute for a longer exposure.
- `range=far`: unambiguous range R = 4 m; `range=near`: R = 2 m (twice the modulation frequency).
  A surface beyond R reads as d − R. `wrap-reflectivity` can remove such pixels, and with
  `qtitofsrc far-reference-interval=N` `qtitofdepth unwrap=true` can output them at their true distance.
- A range change takes effect 40–50 ms after the write. `qtitofsrc` drops the groups that may still be in the old mode
  (`settle-time`).

## Pipelines

```bash
# colour depth on the display (near mode, red = near, blue = far, black = no signal)
gst-launch-1.0 qtitofsrc range=near ! qtitofdepth max-distance=1.5 \
    ! video/x-raw,format=RGBA ! qtivtransform ! video/x-raw,format=RGBA,width=960,height=720 ! waylandsink

# depth in millimetres (240 x 180 GRAY16_LE, 0 = invalid), amplitude and confidence: a queue per branch
gst-launch-1.0 qtitofsrc range=near num-buffers=300 ! qtitofdepth name=d \
    ! queue ! video/x-raw,format=GRAY16_LE ! filesink location=depth.raw \
    d.amplitude ! queue ! video/x-raw,format=GRAY16_LE ! filesink location=amplitude.raw \
    d.confidence ! queue ! video/x-raw,format=GRAY8 ! filesink location=confidence.raw

# record the raw groups (345 664 bytes each), compute depth later with any calibration
# (a replay is timed from the group count at the caps framerate, so it can also feed qtivcomposer)
gst-launch-1.0 qtitofsrc range=near num-buffers=300 ! filesink location=tof.dcs
gst-launch-1.0 filesrc location=tof.dcs blocksize=345664 \
    ! 'video/x-tof-dcs,format=DCS16,width=240,height=180,phases=4,framerate=30/1' \
    ! qtitofdepth ! video/x-raw,format=GRAY16_LE ! filesink location=depth.raw
```

Samples: `gst-sample-apps/gst-qimsdk-tof-depth` (C), `gst-python-examples/gst-tof-depth.py` and
`gst-tof-pointcloud.py` (Python GI). Tests: `gst-test-framework -s tof` (14 cases; three need the camera).

## Depth calibration key file

`qtitofdepth calibration=FILE` takes a GLib key file with one group per range mode, `[far]` and `[near]`. Unset, it
uses `$GST_TOF_CALIBRATION` or the CMake cache variable `TOF_DEFAULT_CALIBRATION`. If that file does not exist the
element does not start, so depth is never silently non-metric. `calibration=""` asks for uncalibrated depth:
R/2π · ((−φ) mod 2π), without the zero-distance phase.

Per pixel, φ = atan2(Q, I) with I = F0 − F2 and Q = F1 − F3. The radial distance d solves

    d = R / 2π · (u + y(d)),   u = (phi_ref − φ) mod 2π,
    y(d) = c0 + c1·d + Σ_k (a_k cos kθ + b_k sin kθ),   θ = 2π d / R

| Key | Meaning |
|---|---|
| `R` | unambiguous range of the mode (m): 4.0 far, 2.0 near |
| `phi_ref` | phase (rad) at zero distance |
| `c0`, `c1` (optional) | constant and linear phase terms (rad, rad/m) |
| `a4`, `b4`, `a8`, `b8`, `a12`, `b12`, `a16`, `b16` (optional) | the cyclic error: harmonics k = 4, 8, 12, 16 of θ (rad) |
| `wrap_k`, `wrap_sigma` (optional) | the reference for `wrap-reflectivity`: amplitude × m² of the reference target, and the per-component noise (counts) |

One way to measure these: record a flat, matte target square-on at known distances, about 0.3–1.1 m in steps of 0.1 m,
in both modes. Fit `phi_ref`, `c0` and the k = 4 terms per mode by least squares on the equation above, and keep one
distance out to test the result. With such a fit one module read a held-out 1.00 m within 1.6 cm in near mode and
0.5 cm in far mode.

## Lens key file

`depth-type=z` and the `pointcloud` output need a lens model: a `[lens]` group in the file `lens=` names, or in the
calibration file.

| Key | Meaning |
|---|---|
| `f` | focal length (pixels) |
| `cx`, `cy` | principal point (pixels; pixel centres are 0-based) |
| `k1`, `k2` (optional) | inverse radial distortion: x = x_d (1 + k1 r_d² + k2 r_d⁴), x_d = (u − cx) / f, likewise y |
| `width`, `height` (optional) | must be 240 and 180 if present |

z = r / |(x, y, 1)|; the point cloud is r · (x, y, 1) / |(x, y, 1)| in metres (x right, y down, z forward), NaN where
the pixel is invalid.

## Buffer format

`video/x-tof-dcs, format=DCS16, width=240, height=180, phases=4, framerate=30/1`: a 64-byte little-endian header
followed by four 240 × 180 planes of int16 (F0..F3, the module's 11-bit two's-complement samples sign-extended, row
stride 480 bytes). `tofdcs.h` has the header layout: magic `TOF1`, flags, group and frame sequence numbers, timestamp,
mode, requested mode and firmware version.
