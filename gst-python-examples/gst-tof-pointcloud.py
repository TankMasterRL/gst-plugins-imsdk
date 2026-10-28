#!/usr/bin/env python3

################################################################################
# Copyright (c) 2026, arducam-tof-ventuno project.
# SPDX-License-Identifier: BSD-3-Clause-Clear
################################################################################

import re
import sys
import time
import signal
import argparse

import numpy as np

import gi
gi.require_version('Gst', '1.0')
gi.require_version("GLib", "2.0")
from gi.repository import Gst, GLib

# Constants
DESCRIPTION = """
This app reads the 3-D point cloud of the Arducam ToF camera
(qtitofsrc ! qtitofdepth, request pad "pointcloud") with the standard
GStreamer Python API and writes it as a PLY file. The pad uses the QIM SDK
tensor format: caps neural-network/tensors, type FLOAT32, dimensions
<1, 180, 240, 3>; every buffer is X, Y, Z in metres per pixel (x right,
y down, z forward, origin at the lens), NaN where there is no valid distance.
The points need a lens model (lens= key file with a [lens] group).
"""

DEPTH_FPS = 30
DCS_CAPS = ("video/x-tof-dcs,format=DCS16,width=240,height=180,phases=4,"
            "framerate=30/1")
DCS_GROUP_BYTES = 345664

waiting_for_eos = False
eos_received = False


class PointCloudState:
    """State shared by the appsink callback and the bus."""

    def __init__(self, average):
        self.average = average
        self.t_start = time.monotonic()
        self.frames = 0
        self.shape = None
        self.last = None        # last frame, (H, W, 3)
        self.sum = None         # per-pixel sum of valid points (--average)
        self.count = None       # per-pixel number of valid frames
        self.first_pts = None
        self.last_pts = None

    def elapsed(self):
        return time.monotonic() - self.t_start


def handle_interrupt_signal(pipeline, mloop):
    """Handle Ctrl+C."""
    global waiting_for_eos

    _, state, _ = pipeline.get_state(Gst.CLOCK_TIME_NONE)
    if state != Gst.State.PLAYING or waiting_for_eos:
        mloop.quit()
        return GLib.SOURCE_CONTINUE

    event = Gst.Event.new_eos()
    if pipeline.send_event(event):
        print("EoS sent to the pipeline")
        waiting_for_eos = True
    else:
        print("Failed to send EoS event to the pipeline!")
        mloop.quit()
    return GLib.SOURCE_CONTINUE


def handle_bus_message(bus, message, mloop):
    """Handle messages posted on pipeline bus."""
    global eos_received

    if message.type == Gst.MessageType.ERROR:
        error, debug_info = message.parse_error()
        print("ERROR:", message.src.get_name(), " ", error.message)
        if debug_info:
            print("debugging info:", debug_info)
        mloop.quit()
    elif message.type == Gst.MessageType.WARNING:
        warning, _ = message.parse_warning()
        print("WARNING:", message.src.get_name(), " ", warning.message)
    elif message.type == Gst.MessageType.EOS:
        eos_received = True
        mloop.quit()
    return True


def tensor_shape(caps):
    """Tensor dimensions from neural-network/tensors caps, e.g. (1, 180, 240, 3)."""
    s = caps.get_structure(0)
    if s.get_name() != "neural-network/tensors" or s.get_string("type") != "FLOAT32":
        raise ValueError(f"unexpected caps {caps.to_string()}")
    dims = re.search(r"dimensions=\(int\)<\s*<([^>]*)>", caps.to_string())
    return tuple(int(v) for v in re.findall(r"\d+", dims.group(1)))


def on_new_sample(appsink, pc):
    """appsink "new-sample": one point cloud."""
    sample = appsink.emit("pull-sample")
    if sample is None:
        return Gst.FlowReturn.ERROR

    if pc.shape is None:
        pc.shape = tensor_shape(sample.get_caps())
        print(f"caps {sample.get_caps().to_string()}", flush=True)
    buffer = sample.get_buffer()
    ok, info = buffer.map(Gst.MapFlags.READ)
    if not ok:
        return Gst.FlowReturn.ERROR
    # (1, H, W, 3) -> (H, W, 3); copy before unmapping
    points = np.frombuffer(info.data, np.float32).reshape(pc.shape)[0].copy()
    buffer.unmap(info)

    valid = ~np.isnan(points[..., 2])
    pc.last = points
    if pc.average:
        if pc.sum is None:
            pc.sum = np.zeros(points.shape, np.float64)
            pc.count = np.zeros(valid.shape, np.int64)
        pc.sum[valid] += points[valid]
        pc.count += valid
    pc.frames += 1
    if buffer.pts != Gst.CLOCK_TIME_NONE:     # a replayed file has no timestamps
        if pc.first_pts is None:
            pc.first_pts = buffer.pts
        pc.last_pts = buffer.pts

    if pc.frames % DEPTH_FPS == 0:
        h, w = valid.shape
        x, y, z = points[h // 2, w // 2]
        centre = (f"({x:+.3f}, {y:+.3f}, {z:.3f}) m" if not np.isnan(z)
                  else "no signal")
        print(f"t={pc.elapsed():5.2f} s  frame {pc.frames:4d}  "
              f"valid {100.0 * valid.mean():5.1f} %  centre point {centre}",
              flush=True)
    return Gst.FlowReturn.OK


def write_ply(path, points):
    """Binary little-endian PLY of the points (N, 3) in metres."""
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\n"
                 f"comment Arducam ToF point cloud, metres, x right, y down, "
                 f"z forward\nelement vertex {len(points)}\n"
                 f"property float x\nproperty float y\nproperty float z\n"
                 f"end_header\n").encode())
        f.write(points.astype("<f4").tobytes())


def final_points(pc):
    """Valid points of the last frame, or with --average the per-pixel mean of
    the pixels valid in at least half of the frames."""
    if pc.average:
        keep = pc.count >= max(1, (pc.frames + 1) // 2)
        return (pc.sum[keep] / pc.count[keep][:, None]).astype(np.float32)
    return pc.last[~np.isnan(pc.last[..., 2])]


def create_element(factory_name, name):
    """Create a GStreamer element."""
    element = Gst.ElementFactory.make(factory_name, name)
    if not element:
        raise Exception(f"Unable to create element {name}")
    return element


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=type(
            'CustomFormatter',
            (argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter),
            {}
        )
    )

    parser.add_argument(
        '-l', '--lens', type=str, default=None,
        help='Lens key file with a [lens] group (format in the gst-plugin-tof '
             'README); default: the [lens] group of the calibration'
    )
    parser.add_argument(
        '-c', '--calibration', type=str, default=None,
        help='Depth calibration key file; default: qtitofdepth\'s '
             '($GST_TOF_CALIBRATION or the path the plugin was built with)'
    )
    parser.add_argument(
        '-f', '--file', type=str, default=None,
        help='Replay a recorded DCS16 file (qtitofsrc ! filesink) instead of the camera'
    )
    parser.add_argument(
        '-r', '--range', type=str, choices=['far', 'near'], default='near',
        help='Range mode of the camera (far: 4 m, near: 2 m unambiguous range)'
    )
    parser.add_argument(
        '-n', '--frames', type=int, default=150,
        help='Point clouds to read from the camera (30 per second)'
    )
    parser.add_argument(
        '-a', '--average', action='store_true',
        help='Write the per-pixel mean over all frames (static scenes; less noise) '
             'instead of the last frame'
    )
    parser.add_argument(
        '-o', '--output', type=str, default='tof-pointcloud.ply',
        help='PLY file to write'
    )

    return parser.parse_args()


def create_pipeline(pipeline, args):
    """Initialize and link elements for the GStreamer pipeline."""
    if args.file:
        source = create_element("filesrc", "source")
        source.set_property("location", args.file)
        source.set_property("blocksize", DCS_GROUP_BYTES)
        dcscaps = create_element("capsfilter", "dcscaps")
        dcscaps.set_property("caps", Gst.Caps.from_string(DCS_CAPS))
        chain = [source, dcscaps]
    else:
        source = create_element("qtitofsrc", "tofsrc")
        source.set_property("range", args.range)
        # num-buffers ends the stream (EOS) after the requested frames
        source.set_property("num-buffers", args.frames)
        chain = [source]

    tofdepth = create_element("qtitofdepth", "tofdepth")
    if args.calibration is not None:
        tofdepth.set_property("calibration", args.calibration)
    if args.lens is not None:
        tofdepth.set_property("lens", args.lens)
    appsink = create_element("appsink", "appsink")
    appsink.set_property("sync", False)
    appsink.set_property("emit-signals", True)

    for element in chain + [tofdepth, appsink]:
        pipeline.add(element)
    for src, dest in zip(chain, chain[1:] + [tofdepth]):
        if not src.link(dest):
            raise Exception(f"Unable to link {src.get_name()} to {dest.get_name()}")
    # The point cloud is a request pad; the depth pad ("src") stays unlinked
    # and is not computed.
    pad = tofdepth.request_pad_simple("pointcloud")
    if not pad or pad.link(appsink.get_static_pad("sink")) != Gst.PadLinkReturn.OK:
        raise Exception("Unable to link tofdepth.pointcloud to the appsink")
    return appsink


def main():
    """Main function to set up and run the GStreamer pipeline."""

    # Initialize GStreamer
    Gst.init(None)
    mloop = GLib.MainLoop()

    # Parse arguments
    args = parse_arguments()

    # Create the pipeline
    try:
        pipeline = Gst.Pipeline.new("tof-pointcloud-pipeline")
        if not pipeline:
            raise Exception("Unable to create ToF point cloud pipeline")
        appsink = create_pipeline(pipeline, args)
    except Exception as e:
        print(f"{e} Exiting...")
        return -1

    pc = PointCloudState(args.average)
    appsink.connect("new-sample", on_new_sample, pc)

    # Handle Ctrl+C
    interrupt_watch_id = GLib.unix_signal_add(
        GLib.PRIORITY_HIGH, signal.SIGINT, handle_interrupt_signal, pipeline, mloop
    )

    # Wait until error or EOS
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", handle_bus_message, mloop)

    # Start playing
    print("Setting to PLAYING...")
    pipeline.set_state(Gst.State.PLAYING)
    mloop.run()

    GLib.source_remove(interrupt_watch_id)
    bus.remove_signal_watch()
    bus = None

    print("Setting to NULL...")
    pipeline.set_state(Gst.State.NULL)

    span_s = ((pc.last_pts or 0) - (pc.first_pts or 0)) / Gst.SECOND
    rate = (f"{(pc.frames - 1) / span_s:.2f} frames/s (PTS)" if span_s > 0
            else "no timestamps (file)")
    print(f"received {pc.frames} point clouds, {rate}")
    if pc.frames:
        points = final_points(pc)
        write_ply(args.output, points)
        print(f"wrote {args.output}: {len(points)} points "
              f"({'mean of ' + str(pc.frames) + ' frames' if args.average else 'last frame'})")

    mloop = None
    pipeline = None
    Gst.deinit()
    if eos_received and pc.frames:
        print("App execution successful")
        return 0
    return -1


if __name__ == "__main__":
    sys.exit(main())
