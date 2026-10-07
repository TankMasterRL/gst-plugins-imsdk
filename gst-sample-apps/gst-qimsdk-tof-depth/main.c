/*
 * Copyright (c) 2026, arducam-tof-ventuno project.
 * SPDX-License-Identifier: BSD-3-Clause-Clear
 *
 * gst-qimsdk-tof-depth: Arducam ToF depth with the standard GStreamer C API (RUN-0055, T5.5).
 *
 *   qtitofsrc -> qtitofdepth -> capsfilter (GRAY16_LE 240x180) -> appsink
 *
 * Reads the distance at the image centre from every depth frame, switches the
 * range mode while the pipeline is PLAYING, and reads the source's controls
 * and statistics back.
 */

#include <stdio.h>
#include <string.h>
#include <glib-unix.h>
#include <gst/gst.h>
#include <gst/video/video.h>
#include <glib.h>

#include <gst/sampleapps/gst_sample_apps_utils.h>

/* Run length, range modes and the moment of the switch (30 groups/s). */
#define DURATION_S      10
#define START_RANGE     "far"   /* far: unambiguous range 4 m; near: 2 m */
#define SWITCH_RANGE    "near"
#define SWITCH_AT_S     5

/* Depth calibration (key file of tools/tof-sweep.py, calibration v1); without it
 * qtitofdepth outputs uncorrected, non-metric depth. Device-specific absolute path. */
#define CALIBRATION_FILE "/home/arduino/Downloads/arducam-tof-ventuno/calib/tof-depth-cal-v1.conf"

/* qtitofdepth output: 240x180 GRAY16_LE, value = distance in mm, 0 = invalid. */
#define DEPTH_WIDTH     240
#define DEPTH_HEIGHT    180
#define DEPTH_FPS       30
#define CENTRE_HALF     2       /* 5x5 window around the centre pixel */

/* Application state shared by the appsink callbacks and the range timer. */
typedef struct {
  GstElement *tofsrc;
  gint64 t_start;
  guint frames;
  guint invalid;
  GstClockTime first_pts;
  GstClockTime last_pts;
} TofDepthState;

static gdouble
elapsed_s (TofDepthState * state)
{
  return (g_get_monotonic_time () - state->t_start) / 1e6;
}

/* Nick of the current value of an enum property, e.g. "far". */
static const gchar *
enum_property_nick (GstElement * element, const gchar * name)
{
  GParamSpec *pspec;
  GEnumValue *value;
  gint current = 0;

  pspec = g_object_class_find_property (G_OBJECT_GET_CLASS (element), name);
  if (pspec == NULL || !G_IS_PARAM_SPEC_ENUM (pspec))
    return "?";

  g_object_get (element, name, &current, NULL);
  value = g_enum_get_value (G_PARAM_SPEC_ENUM (pspec)->enum_class, current);
  return value != NULL ? value->value_nick : "?";
}

/* Median of the valid (non-zero) pixels in the centre window; 0 if none. */
static guint
centre_distance_mm (const guint8 * data)
{
  guint16 values[(2 * CENTRE_HALF + 1) * (2 * CENTRE_HALF + 1)];
  guint n = 0, i, j;
  gint x, y;

  for (y = DEPTH_HEIGHT / 2 - CENTRE_HALF; y <= DEPTH_HEIGHT / 2 + CENTRE_HALF; y++) {
    for (x = DEPTH_WIDTH / 2 - CENTRE_HALF; x <= DEPTH_WIDTH / 2 + CENTRE_HALF; x++) {
      guint16 mm;

      memcpy (&mm, data + 2 * (y * DEPTH_WIDTH + x), sizeof (mm));
      if (mm == 0)
        continue;

      /* Insertion sort: the window has at most 25 values. */
      for (i = n; i > 0 && values[i - 1] > mm; i--)
        values[i] = values[i - 1];
      values[i] = mm;
      n++;
    }
  }

  j = n / 2;
  return n > 0 ? values[j] : 0;
}

/* appsink "new-sample": one depth frame. */
static GstFlowReturn
on_new_sample (GstElement * appsink, gpointer userdata)
{
  TofDepthState *state = (TofDepthState *) userdata;
  GstSample *sample = NULL;
  GstBuffer *buffer;
  GstMapInfo map;
  guint distance;

  g_signal_emit_by_name (appsink, "pull-sample", &sample);
  if (sample == NULL)
    return GST_FLOW_ERROR;

  buffer = gst_sample_get_buffer (sample);
  if (buffer == NULL || !gst_buffer_map (buffer, &map, GST_MAP_READ)) {
    gst_sample_unref (sample);
    return GST_FLOW_ERROR;
  }

  if (map.size != DEPTH_WIDTH * DEPTH_HEIGHT * 2) {
    gst_buffer_unmap (buffer, &map);
    gst_sample_unref (sample);
    return GST_FLOW_OK;
  }

  distance = centre_distance_mm (map.data);
  gst_buffer_unmap (buffer, &map);

  state->frames++;
  if (state->frames == 1)
    state->first_pts = GST_BUFFER_PTS (buffer);
  state->last_pts = GST_BUFFER_PTS (buffer);
  if (distance == 0)
    state->invalid++;

  if (state->frames == 1) {
    guint firmware = 0, device = 0;

    g_object_get (state->tofsrc, "firmware-version", &firmware,
        "device-id", &device, NULL);
    g_print ("module firmware 0x%x, device id 0x%04x\n", firmware, device);
  }

  if (state->frames % DEPTH_FPS == 0) {
    if (distance > 0)
      g_print ("t=%5.2f s  frame %4u  range %-4s  centre %u mm\n",
          elapsed_s (state), state->frames,
          enum_property_nick (state->tofsrc, "range"), distance);
    else
      g_print ("t=%5.2f s  frame %4u  range %-4s  centre no signal\n",
          elapsed_s (state), state->frames,
          enum_property_nick (state->tofsrc, "range"));
  }

  gst_sample_unref (sample);
  return GST_FLOW_OK;
}

/* appsink "eos": print the source statistics before the pipeline stops. */
static void
on_eos (GstElement * appsink, gpointer userdata)
{
  TofDepthState *state = (TofDepthState *) userdata;
  GstStructure *stats = NULL;
  gchar *text;

  g_object_get (state->tofsrc, "stats", &stats, NULL);
  if (stats == NULL)
    return;

  text = gst_structure_to_string (stats);
  g_print ("EOS: %s\n", text);
  g_free (text);
  gst_structure_free (stats);
}

/* Main-loop timeout: switch the range mode while the pipeline is PLAYING. */
static gboolean
switch_range (gpointer userdata)
{
  TofDepthState *state = (TofDepthState *) userdata;

  gst_element_set_enum_property (state->tofsrc, "range", SWITCH_RANGE);
  g_print ("t=%5.2f s  range set to %s\n", elapsed_s (state), SWITCH_RANGE);
  return G_SOURCE_REMOVE;
}

static gboolean
create_pipe (GstAppContext * appctx, TofDepthState * state)
{
  GstElement *tofsrc = NULL;
  GstElement *tofdepth = NULL;
  GstElement *depth_caps = NULL;
  GstElement *appsink = NULL;
  GstCaps *caps = NULL;

  /* Create all elements */
  /* ToF source from the gst-plugins-imsdk fork: sets up the CAMSS path itself */
  tofsrc = gst_element_factory_make ("qtitofsrc", "tofsrc");
  if (!tofsrc) {
    g_printerr ("Failed to create qtitofsrc\n");
    goto cleanup;
  }

  /* Calibrated depth from the DCS groups */
  tofdepth = gst_element_factory_make ("qtitofdepth", "tofdepth");
  if (!tofdepth) {
    g_printerr ("Failed to create qtitofdepth\n");
    goto cleanup;
  }

  /* Selects the metric output of qtitofdepth (millimetres) */
  depth_caps = gst_element_factory_make ("capsfilter", "depth_caps");
  if (!depth_caps) {
    g_printerr ("Failed to create capsfilter\n");
    goto cleanup;
  }

  /* Application sink: receives each depth frame in on_new_sample() */
  appsink = gst_element_factory_make ("appsink", "appsink");
  if (!appsink) {
    g_printerr ("Failed to create appsink\n");
    goto cleanup;
  }

  /* Set properties */
  gst_element_set_enum_property (tofsrc, "range", START_RANGE);
  /* num-buffers ends the stream (EOS) after DURATION_S seconds of groups */
  g_object_set (G_OBJECT (tofsrc), "num-buffers", DURATION_S * DEPTH_FPS, NULL);
  g_object_set (G_OBJECT (tofdepth), "calibration", CALIBRATION_FILE, NULL);

  caps = gst_caps_new_simple ("video/x-raw",
      "format", G_TYPE_STRING, "GRAY16_LE",
      "width", G_TYPE_INT, DEPTH_WIDTH,
      "height", G_TYPE_INT, DEPTH_HEIGHT,
      "framerate", GST_TYPE_FRACTION, DEPTH_FPS, 1, NULL);
  g_object_set (G_OBJECT (depth_caps), "caps", caps, NULL);
  gst_caps_unref (caps);
  caps = NULL;

  g_object_set (G_OBJECT (appsink), "sync", FALSE, "emit-signals", TRUE, NULL);

  /* Add all elements to the pipeline */
  gst_bin_add_many (GST_BIN (appctx->pipeline), tofsrc, tofdepth, depth_caps,
      appsink, NULL);

  /* Link elements */
  if (!gst_element_link_many (tofsrc, tofdepth, depth_caps, appsink, NULL)) {
    g_printerr ("Failed to link pipeline\n");
    goto cleanup_pipeline;
  }

  state->tofsrc = tofsrc;
  g_signal_connect (appsink, "new-sample", G_CALLBACK (on_new_sample), state);
  g_signal_connect (appsink, "eos", G_CALLBACK (on_eos), state);

  return TRUE;

cleanup_pipeline:
  gst_object_unref (appctx->pipeline);
  appctx->pipeline = NULL;
  return FALSE;

cleanup:
  /* Unref only elements not yet added to the pipeline */
  if (tofsrc)
    gst_object_unref (tofsrc);
  if (tofdepth)
    gst_object_unref (tofdepth);
  if (depth_caps)
    gst_object_unref (depth_caps);
  if (appsink)
    gst_object_unref (appsink);
  if (caps)
    gst_caps_unref (caps);
  return FALSE;
}

gint
main (gint argc, gchar * argv[])
{
  GstAppContext appctx = {};
  TofDepthState state = {};
  GstBus *bus = NULL;
  guint intrpt_watch_id = 0;
  guint term_watch_id = 0;
  gdouble span_s;
  gint ret = 0;

  /* Initialize GStreamer */
  gst_init (&argc, &argv);
  state.t_start = g_get_monotonic_time ();

  /* Create pipeline container */
  appctx.pipeline = gst_pipeline_new ("gst-qimsdk-tof-depth");
  if (!appctx.pipeline) {
    g_printerr ("Failed to create pipeline\n");
    ret = -1;
    goto done;
  }

  /* Create main loop */
  appctx.mloop = g_main_loop_new (NULL, FALSE);
  if (!appctx.mloop) {
    g_printerr ("Failed to create main loop\n");
    ret = -1;
    goto done;
  }

  /* Build pipeline */
  if (!create_pipe (&appctx, &state)) {
    g_printerr ("Failed to build pipeline\n");
    ret = -1;
    goto done;
  }

  /* Set up bus — callbacks from gst_sample_apps_utils.h */
  bus = gst_pipeline_get_bus (GST_PIPELINE (appctx.pipeline));
  gst_bus_add_signal_watch (bus);
  g_signal_connect (bus, "message::error", G_CALLBACK (error_cb), appctx.mloop);
  g_signal_connect (bus, "message::warning", G_CALLBACK (warning_cb), NULL);
  g_signal_connect (bus, "message::eos", G_CALLBACK (eos_cb), appctx.mloop);
  g_signal_connect (bus, "message::state-changed", G_CALLBACK (state_changed_cb),
      appctx.pipeline);
  gst_object_unref (bus);

  /* Interrupt handlers — from gst_sample_apps_utils.h */
  intrpt_watch_id = g_unix_signal_add (SIGINT, handle_interrupt_signal, &appctx);
  term_watch_id = g_unix_signal_add (SIGTERM, handle_interrupt_signal, &appctx);

  /* Range switch while PLAYING */
  g_timeout_add_seconds (SWITCH_AT_S, switch_range, &state);

  /* PAUSED first; state_changed_cb moves the pipeline to PLAYING */
  switch (gst_element_set_state (appctx.pipeline, GST_STATE_PAUSED)) {
    case GST_STATE_CHANGE_FAILURE:
      g_printerr ("Failed to set pipeline to PAUSED\n");
      ret = -1;
      goto done;
    case GST_STATE_CHANGE_NO_PREROLL:
      /* Live source: no preroll, go to PLAYING directly */
      gst_element_set_state (appctx.pipeline, GST_STATE_PLAYING);
      break;
    case GST_STATE_CHANGE_ASYNC:
    case GST_STATE_CHANGE_SUCCESS:
      break;
  }

  g_main_loop_run (appctx.mloop);

  span_s = (state.frames > 1) ?
      (gdouble) (state.last_pts - state.first_pts) / GST_SECOND : 0.0;
  g_print ("received %u depth frames, %.2f frames/s (PTS), %u without a valid "
      "centre\n", state.frames, span_s > 0 ? (state.frames - 1) / span_s : 0.0,
      state.invalid);

done:
  if (intrpt_watch_id)
    g_source_remove (intrpt_watch_id);
  if (term_watch_id)
    g_source_remove (term_watch_id);

  if (appctx.pipeline) {
    gst_element_set_state (appctx.pipeline, GST_STATE_NULL);
    gst_object_unref (appctx.pipeline);
  }
  if (appctx.mloop)
    g_main_loop_unref (appctx.mloop);

  gst_deinit ();
  return ret;
}
