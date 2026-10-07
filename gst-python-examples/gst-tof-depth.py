#!/usr/bin/env python3

################################################################################
# Copyright (c) 2026, arducam-tof-ventuno project.
# SPDX-License-Identifier: BSD-3-Clause-Clear
################################################################################

import os
import sys
import time
import signal
import struct
import argparse

import gi
gi.require_version('Gst', '1.0')
gi.require_version("GLib", "2.0")
from gi.repository import Gst, GLib

# Constants
DESCRIPTION = """
This app shows live depth from the Arducam ToF camera (qtitofsrc ! qtitofdepth)
with the standard GStreamer Python API: it prints the distance at the image
centre once per second, switches the range mode while the pipeline is PLAYING,
and reads the source's controls and statistics back.
"""
DEFAULT_CALIBRATION = os.path.join(
    os.path.expanduser("~"),
    "Downloads/arducam-tof-ventuno/calib/tof-depth-cal-v1.conf")

# qtitofdepth output: 240x180 GRAY16_LE, value = distance in mm, 0 = invalid.
DEPTH_WIDTH = 240
DEPTH_HEIGHT = 180
DEPTH_FPS = 30
CENTRE_HALF = 2  # 5x5 window around the centre pixel

waiting_for_eos = False
eos_received = False


class TofDepthState:
    """State shared by the appsink callback, the range timer and the bus."""

    def __init__(self, tofsrc):
        self.tofsrc = tofsrc
        self.t_start = time.monotonic()
        self.frames = 0
        self.invalid = 0
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


def handle_bus_message(bus, message, mloop, tof):
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
        # Read the source statistics before the pipeline stops.
        print("EOS:", tof.tofsrc.get_property("stats").to_string())
        eos_received = True
        mloop.quit()
    return True


def centre_distance_mm(buffer):
    """Median of the valid (non-zero) pixels in the centre window, or None."""
    row_bytes = DEPTH_WIDTH * 2
    first_row = DEPTH_HEIGHT // 2 - CENTRE_HALF
    rows = buffer.extract_dup(first_row * row_bytes,
                              (2 * CENTRE_HALF + 1) * row_bytes)
    pixels = struct.unpack(f"<{len(rows) // 2}H", rows)
    cx = DEPTH_WIDTH // 2
    values = sorted(
        pixels[r * DEPTH_WIDTH + x]
        for r in range(2 * CENTRE_HALF + 1)
        for x in range(cx - CENTRE_HALF, cx + CENTRE_HALF + 1)
        if pixels[r * DEPTH_WIDTH + x] != 0
    )
    return values[len(values) // 2] if values else None


def on_new_sample(appsink, tof):
    """appsink "new-sample": one depth frame."""
    sample = appsink.emit("pull-sample")
    if sample is None:
        return Gst.FlowReturn.ERROR

    buffer = sample.get_buffer()
    if buffer.get_size() != DEPTH_WIDTH * DEPTH_HEIGHT * 2:
        return Gst.FlowReturn.OK

    distance = centre_distance_mm(buffer)
    tof.frames += 1
    if tof.first_pts is None:
        tof.first_pts = buffer.pts
    tof.last_pts = buffer.pts
    if distance is None:
        tof.invalid += 1

    if tof.frames == 1:
        print(f"module firmware 0x{tof.tofsrc.get_property('firmware-version'):x}, "
              f"device id 0x{tof.tofsrc.get_property('device-id'):04x}", flush=True)

    if tof.frames % DEPTH_FPS == 0:
        mode = tof.tofsrc.get_property("range").value_nick
        shown = f"{distance} mm" if distance is not None else "no signal"
        print(f"t={tof.elapsed():5.2f} s  frame {tof.frames:4d}  range {mode:4s}  "
              f"centre {shown}", flush=True)
    return Gst.FlowReturn.OK


def switch_range(tof, mode):
    """Main-loop timeout: switch the range mode while the pipeline is PLAYING."""
    tof.tofsrc.set_property("range", mode)
    print(f"t={tof.elapsed():5.2f} s  range set to {mode}", flush=True)
    return GLib.SOURCE_REMOVE


def create_element(factory_name, name):
    """Create a GStreamer element."""
    element = Gst.ElementFactory.make(factory_name, name)
    if not element:
        raise Exception(f"Unable to create element {name}")
    return element


def link_elements(link_order, elements):
    """Link elements in the specified order."""
    src = None
    for element in link_order:
        dest = elements[element]
        if src and not src.link(dest):
            raise Exception(
                f"Unable to link element {src.get_name()} to "
                f"{dest.get_name()}"
            )
        src = dest


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
        '-r', '--range', type=str, choices=['far', 'near'], default='far',
        help='Range mode at start (far: 4 m, near: 2 m unambiguous range)'
    )
    parser.add_argument(
        '-s', '--switch-to', type=str, choices=['far', 'near'], default='near',
        help='Range mode to switch to while playing'
    )
    parser.add_argument(
        '-t', '--switch-after', type=int, default=5,
        help='Seconds after start of the range switch (0: no switch)'
    )
    parser.add_argument(
        '-d', '--duration', type=int, default=10,
        help='Run length in seconds (30 depth frames per second)'
    )
    parser.add_argument(
        '-c', '--calibration', type=str, default=DEFAULT_CALIBRATION,
        help='Depth calibration key file (tools/tof-sweep.py); without it depth is not metric'
    )

    return parser.parse_args()


def create_pipeline(pipeline, args):
    """Initialize and link elements for the GStreamer pipeline."""
    # Create elements
    elements = {
        "tofsrc"  : create_element("qtitofsrc", "tofsrc"),
        "tofdepth": create_element("qtitofdepth", "tofdepth"),
        "depthcaps": create_element("capsfilter", "depthcaps"),
        "appsink" : create_element("appsink", "appsink"),
    }

    # Set properties
    elements["tofsrc"].set_property("range", args.range)
    # num-buffers ends the stream (EOS) after the run length
    elements["tofsrc"].set_property("num-buffers", args.duration * DEPTH_FPS)
    elements["tofdepth"].set_property("calibration", args.calibration)
    # Metric output of qtitofdepth (millimetres)
    elements["depthcaps"].set_property(
        "caps", Gst.Caps.from_string(
            "video/x-raw,format=GRAY16_LE,"
            f"width={DEPTH_WIDTH},height={DEPTH_HEIGHT},"
            f"framerate={DEPTH_FPS}/1"
        )
    )
    elements["appsink"].set_property("sync", False)
    elements["appsink"].set_property("emit-signals", True)

    # Add elements to the pipeline
    for element in elements.values():
        pipeline.add(element)

    # Link elements
    link_order = ["tofsrc", "tofdepth", "depthcaps", "appsink"]
    link_elements(link_order, elements)

    return elements


def main():
    """Main function to set up and run the GStreamer pipeline."""

    # Initialize GStreamer
    Gst.init(None)
    mloop = GLib.MainLoop()

    # Parse arguments
    args = parse_arguments()

    # Create the pipeline
    try:
        pipeline = Gst.Pipeline.new("tof-depth-pipeline")
        if not pipeline:
            raise Exception("Unable to create ToF depth pipeline")
        elements = create_pipeline(pipeline, args)
    except Exception as e:
        print(f"{e} Exiting...")
        return -1

    tof = TofDepthState(elements["tofsrc"])
    elements["appsink"].connect("new-sample", on_new_sample, tof)

    # Handle Ctrl+C
    interrupt_watch_id = GLib.unix_signal_add(
        GLib.PRIORITY_HIGH, signal.SIGINT, handle_interrupt_signal, pipeline, mloop
    )

    # Range switch while playing
    if args.switch_after > 0:
        GLib.timeout_add_seconds(args.switch_after, switch_range, tof, args.switch_to)

    # Wait until error or EOS
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", handle_bus_message, mloop, tof)

    # Start playing
    print("Setting to PLAYING...")
    pipeline.set_state(Gst.State.PLAYING)
    mloop.run()

    GLib.source_remove(interrupt_watch_id)
    bus.remove_signal_watch()
    bus = None

    print("Setting to NULL...")
    pipeline.set_state(Gst.State.NULL)

    span_s = ((tof.last_pts or 0) - (tof.first_pts or 0)) / Gst.SECOND
    rate = (tof.frames - 1) / span_s if span_s > 0 else 0.0
    print(f"received {tof.frames} depth frames, {rate:.2f} frames/s (PTS), "
          f"{tof.invalid} without a valid centre")

    mloop = None
    pipeline = None
    Gst.deinit()
    if eos_received:
        print("App execution successful")
        return 0
    return -1


if __name__ == "__main__":
    sys.exit(main())
