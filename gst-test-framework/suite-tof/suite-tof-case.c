/*
 * Copyright (c) 2026, arducam-tof-ventuno project.
 * SPDX-License-Identifier: BSD-3-Clause-Clear
 *
 * ToF suite (RUN-0050, T5.5): qtitofsrc and qtitofdepth (gst-plugin-tof, DEC-0020).
 *   elements      factories, properties and pad templates
 *   stream        live qtitofsrc for the run time: DCS16 headers, consecutive groups, PTS, stats
 *   range-switch  range far -> near in PLAYING: header mode follows, settle drops counted
 *   depth         qtitofdepth on synthetic groups (known phase per pixel) vs the analytic distance,
 *                 uncalibrated (near) and with a pure-offset calibration key file (far)
 * The live cases need the Arducam ToF on CAMERA0 with the ToF boot entry and drivers loaded.
 */

#include <math.h>
#include <string.h>
#include <unistd.h>
#include <glib/gstdio.h>

#include "plugin-suite.h"

#define TOF_W          240
#define TOF_H          180
#define TOF_PIX        (TOF_W * TOF_H)
#define TOF_HDR        64
#define TOF_SIZE       (TOF_HDR + 4 * TOF_PIX * 2)
#define TOF_MAGIC      0x31464f54
#define TOF_DCS_CAPS   "video/x-tof-dcs, format=(string)DCS16, width=(int)240, " \
                       "height=(int)180, phases=(int)4, framerate=(fraction)30/1"

static gint runningtime = 10;

typedef struct
{
  GArray *mode;                 /* guint8 */
  GArray *gseq;                 /* guint32 */
  GArray *pts;                  /* guint64 */
  guint bad_header;
  guint bad_frame_seq;
  guint32 firmware;
  GstElement *src;
  guint switch_at;
} TofRecord;

static GstPadProbeReturn
record_probe (GstPad * pad, GstPadProbeInfo * info, gpointer data)
{
  TofRecord *rec = data;
  GstBuffer *buf = GST_PAD_PROBE_INFO_BUFFER (info);
  GstMapInfo map;

  if (!gst_buffer_map (buf, &map, GST_MAP_READ))
    return GST_PAD_PROBE_OK;
  if (map.size != TOF_SIZE || GST_READ_UINT32_LE (map.data) != TOF_MAGIC ||
      GST_READ_UINT16_LE (map.data + 4) != TOF_HDR) {
    rec->bad_header++;
  } else {
    guint8 mode = map.data[40];
    guint32 gseq = GST_READ_UINT32_LE (map.data + 8);
    guint64 pts = GST_BUFFER_PTS (buf);

    for (guint k = 1; k < 4; k++)
      if (GST_READ_UINT32_LE (map.data + 12 + 4 * k) !=
          GST_READ_UINT32_LE (map.data + 12 + 4 * (k - 1)) + 1)
        rec->bad_frame_seq++;
    rec->firmware = GST_READ_UINT32_LE (map.data + 42);
    g_array_append_val (rec->mode, mode);
    g_array_append_val (rec->gseq, gseq);
    g_array_append_val (rec->pts, pts);
  }
  gst_buffer_unmap (buf, &map);
  if (rec->switch_at && rec->mode->len == rec->switch_at)
    gst_util_set_object_arg (G_OBJECT (rec->src), "range", "near");
  return GST_PAD_PROBE_OK;
}

static void
run_until_eos (GstElement * pipeline, guint timeout_s)
{
  GstBus *bus = gst_element_get_bus (pipeline);
  GstMessage *msg;

  fail_unless (gst_element_set_state (pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE);
  msg = gst_bus_timed_pop_filtered (bus, timeout_s * GST_SECOND,
      GST_MESSAGE_EOS | GST_MESSAGE_ERROR);
  fail_unless (msg != NULL, "no EOS within %u s", timeout_s);
  if (GST_MESSAGE_TYPE (msg) == GST_MESSAGE_ERROR) {
    GError *err = NULL;
    gchar *dbg = NULL;

    gst_message_parse_error (msg, &err, &dbg);
    fail ("pipeline error: %s (%s)", err->message, GST_STR_NULL (dbg));
  }
  gst_message_unref (msg);
  gst_object_unref (bus);
}

static void
stream_and_record (const gchar * desc, guint groups, guint switch_at, TofRecord * rec,
    GstStructure ** stats)
{
  GError *err = NULL;
  GstElement *pipeline = gst_parse_launch (desc, &err);
  GstPad *pad;

  fail_unless (pipeline != NULL, "parse: %s", err ? err->message : "?");
  memset (rec, 0, sizeof (*rec));
  rec->mode = g_array_new (FALSE, FALSE, sizeof (guint8));
  rec->gseq = g_array_new (FALSE, FALSE, sizeof (guint32));
  rec->pts = g_array_new (FALSE, FALSE, sizeof (guint64));
  rec->src = gst_bin_get_by_name (GST_BIN (pipeline), "s");
  rec->switch_at = switch_at;
  fail_unless (rec->src != NULL);
  pad = gst_element_get_static_pad (rec->src, "src");
  gst_pad_add_probe (pad, GST_PAD_PROBE_TYPE_BUFFER, record_probe, rec, NULL);
  gst_object_unref (pad);

  run_until_eos (pipeline, groups / 30 + 15);
  g_object_get (rec->src, "stats", stats, NULL);
  gst_element_set_state (pipeline, GST_STATE_NULL);
  gst_object_unref (rec->src);
  gst_object_unref (pipeline);
}

static void
record_free (TofRecord * rec)
{
  g_array_unref (rec->mode);
  g_array_unref (rec->gseq);
  g_array_unref (rec->pts);
}

GST_START_TEST (test_tof_elements)
{
  const gchar *src_props[] = { "range", "settle-time", "exposure", "frame-rate",
    "firmware-version", "device-id", "mux-gpio", "csiphy", "setup-pipeline", "stats", NULL };
  const gchar *depth_props[] = { "calibration", "min-amplitude", "min-distance", "max-distance",
    "colormap", "amplitude-max", "confidence-scale", "temporal-average", "stats", NULL };
  GstElement *src = gst_element_factory_make ("qtitofsrc", NULL);
  GstElement *depth = gst_element_factory_make ("qtitofdepth", NULL);
  GstPad *amp, *conf;

  fail_unless (src != NULL, "qtitofsrc not registered");
  fail_unless (depth != NULL, "qtitofdepth not registered");
  for (guint i = 0; src_props[i]; i++)
    fail_unless (g_object_class_find_property (G_OBJECT_GET_CLASS (src), src_props[i]) != NULL,
        "qtitofsrc lacks %s", src_props[i]);
  for (guint i = 0; depth_props[i]; i++)
    fail_unless (g_object_class_find_property (G_OBJECT_GET_CLASS (depth), depth_props[i]) != NULL,
        "qtitofdepth lacks %s", depth_props[i]);
  amp = gst_element_request_pad_simple (depth, "amplitude");
  conf = gst_element_request_pad_simple (depth, "confidence");
  fail_unless (amp != NULL && conf != NULL, "request pads amplitude/confidence");
  gst_element_release_request_pad (depth, amp);
  gst_element_release_request_pad (depth, conf);
  gst_object_unref (amp);
  gst_object_unref (conf);
  gst_object_unref (src);
  gst_object_unref (depth);
}
GST_END_TEST;

GST_START_TEST (test_tof_stream)
{
  guint n = runningtime * 30;
  gchar *desc = g_strdup_printf ("qtitofsrc name=s range=near num-buffers=%u ! fakesink sync=false", n);
  TofRecord rec;
  GstStructure *stats = NULL;
  guint64 gaps = 1;
  guint steps_bad = 0, pts_bad = 0;

  stream_and_record (desc, n, 0, &rec, &stats);
  g_free (desc);
  fail_unless_equals_int (rec.mode->len, n);
  fail_unless_equals_int (rec.bad_header, 0);
  fail_unless_equals_int (rec.bad_frame_seq, 0);
  for (guint i = 1; i < rec.gseq->len; i++) {
    if (g_array_index (rec.gseq, guint32, i) != g_array_index (rec.gseq, guint32, i - 1) + 1)
      steps_bad++;
    if (g_array_index (rec.pts, guint64, i) <= g_array_index (rec.pts, guint64, i - 1))
      pts_bad++;
  }
  fail_unless_equals_int (steps_bad, 0);
  fail_unless_equals_int (pts_bad, 0);
  for (guint i = 0; i < rec.mode->len; i++)
    fail_unless_equals_int (g_array_index (rec.mode, guint8, i), 1);
  fail_unless (rec.firmware != 0, "firmware version missing in the header");
  fail_unless (gst_structure_get_uint64 (stats, "sequence-gaps", &gaps));
  fail_unless_equals_uint64 (gaps, 0);
  {
    gdouble span = (g_array_index (rec.pts, guint64, n - 1) - g_array_index (rec.pts, guint64, 0)) / 1e9;
    gdouble rate = (n - 1) / span;

    fail_unless (fabs (rate - 30.0) < 1.5, "rate %.3f groups/s", rate);
  }
  gst_structure_free (stats);
  record_free (&rec);
}
GST_END_TEST;

GST_START_TEST (test_tof_range_switch)
{
  TofRecord rec;
  GstStructure *stats = NULL;
  guint64 switches = 0, dropped = 0;
  guint first_near = G_MAXUINT;

  stream_and_record ("qtitofsrc name=s range=far num-buffers=150 ! fakesink sync=false",
      150, 60, &rec, &stats);
  fail_unless_equals_int (rec.bad_header, 0);
  for (guint i = 0; i < rec.mode->len; i++) {
    guint8 m = g_array_index (rec.mode, guint8, i);

    if (i < 60)
      fail_unless_equals_int (m, 0);
    if (m == 1 && first_near == G_MAXUINT)
      first_near = i;
    if (first_near != G_MAXUINT)
      fail_unless_equals_int (m, 1);
  }
  fail_unless (first_near != G_MAXUINT, "no near group after the switch");
  fail_unless (gst_structure_get_uint64 (stats, "range-switches", &switches));
  fail_unless (gst_structure_get_uint64 (stats, "groups-dropped-switch", &dropped));
  fail_unless_equals_uint64 (switches, 1);
  /* settle-time 60 ms with groups every 33 ms: 1-3 groups start inside it */
  fail_unless (dropped >= 1 && dropped <= 3, "dropped %" G_GUINT64_FORMAT, dropped);
  gst_structure_free (stats);
  record_free (&rec);
}
GST_END_TEST;

/* One synthetic group: per pixel phase phi = -pi + 2 pi (p + 0.5) / N, F0 = -F2 = round (A cos phi),
 * F1 = -F3 = round (A sin phi), so I = 2 F0, Q = 2 F1. */
static GstBuffer *
synthetic_group (guint8 mode, gdouble amp)
{
  GstBuffer *buf = gst_buffer_new_allocate (NULL, TOF_SIZE, NULL);
  GstMapInfo map;
  gint16 *f;

  gst_buffer_map (buf, &map, GST_MAP_WRITE);
  memset (map.data, 0, TOF_HDR);
  GST_WRITE_UINT32_LE (map.data, TOF_MAGIC);
  GST_WRITE_UINT16_LE (map.data + 4, TOF_HDR);
  map.data[40] = map.data[41] = mode;
  f = (gint16 *) (map.data + TOF_HDR);
  for (guint p = 0; p < TOF_PIX; p++) {
    gdouble phi = -G_PI + 2.0 * G_PI * (p + 0.5) / TOF_PIX;
    gint16 c = (gint16) lround (amp * cos (phi)), s = (gint16) lround (amp * sin (phi));

    f[p] = GINT16_TO_LE (c);
    f[TOF_PIX + p] = GINT16_TO_LE (s);
    f[2 * TOF_PIX + p] = GINT16_TO_LE (-c);
    f[3 * TOF_PIX + p] = GINT16_TO_LE (-s);
  }
  gst_buffer_unmap (buf, &map);
  return buf;
}

/* depth output (GRAY16_LE mm) vs d = R wrap (phi_ref - phi + c0) / 2pi from the actual I/Q */
static void
check_depth (const gchar * launch, guint8 mode, gdouble r, gdouble phi_ref, gdouble c0)
{
  GstHarness *h = gst_harness_new_parse (launch);
  GstBuffer *in = synthetic_group (mode, 200.0), *out;
  GstMapInfo mi, mo;
  guint checked = 0, worst_px = 0;
  gint worst = 0;

  gst_harness_set_src_caps_str (h, TOF_DCS_CAPS);
  gst_harness_set_sink_caps_str (h,
      "video/x-raw, format=(string)GRAY16_LE, width=(int)240, height=(int)180, framerate=(fraction)30/1");
  fail_unless_equals_int (gst_harness_push (h, gst_buffer_ref (in)), GST_FLOW_OK);
  out = gst_harness_pull (h);
  fail_unless (out != NULL);
  gst_buffer_map (in, &mi, GST_MAP_READ);
  gst_buffer_map (out, &mo, GST_MAP_READ);
  fail_unless (mo.size >= TOF_PIX * 2);
  for (guint p = 0; p < TOF_PIX; p++) {
    const gint16 *f = (const gint16 *) (mi.data + TOF_HDR);
    gdouble i = 2.0 * GINT16_FROM_LE (f[p]), q = 2.0 * GINT16_FROM_LE (f[TOF_PIX + p]);
    gdouble u = phi_ref - atan2 (q, i) + c0, d;
    gint e, got = GST_READ_UINT16_LE (mo.data + 2 * p);

    u -= 2.0 * G_PI * floor (u / (2.0 * G_PI));
    d = r * u / (2.0 * G_PI);
    if (d < 0.002 || d > r - 0.002)
      continue;                 /* at the wrap, 0 and R are the same distance */
    e = got - (gint) lround (1000.0 * d);
    if (ABS (e) > ABS (worst)) {
      worst = e;
      worst_px = p;
    }
    checked++;
  }
  gst_buffer_unmap (in, &mi);
  gst_buffer_unmap (out, &mo);
  fail_unless (checked > TOF_PIX * 0.99, "checked %u pixels", checked);
  fail_unless (ABS (worst) <= 1, "max error %d mm at pixel %u", worst, worst_px);
  gst_buffer_unref (in);
  gst_buffer_unref (out);
  gst_harness_teardown (h);
}

GST_START_TEST (test_tof_depth_synthetic)
{
  gchar *path = NULL, *launch;
  gint fd = g_file_open_tmp ("tof-cal-XXXXXX.conf", &path, NULL);
  const gchar *cal = "[calibration]\nformat=2\n"
      "[far]\nR=4.0\nphi_ref=1.234\nc0=0.1\n[near]\nR=2.0\nphi_ref=-0.5\nc0=-0.05\n";

  /* uncalibrated: phi_ref 0, no correction, near mode R = 2 m */
  check_depth ("qtitofdepth", 1, 2.0, 0.0, 0.0);

  fail_unless (fd >= 0);
  fail_unless (g_file_set_contents (path, cal, -1, NULL));
  close (fd);
  launch = g_strdup_printf ("qtitofdepth calibration=%s", path);
  check_depth (launch, 0, 4.0, 1.234, 0.1);
  check_depth (launch, 1, 2.0, -0.5, -0.05);
  g_free (launch);
  g_unlink (path);
  g_free (path);
}
GST_END_TEST;

static Suite *
tof_suite (GList ** tcnames, gint iteration, gint duration)
{
  Suite *s = suite_create ("tof");
  TCase *tc;
  const gchar *tcname;
  gint start = 0, end = iteration;
  guint tctimeout = duration + 30;

  runningtime = duration;

  tcname = "elements";
  tc = tcase_create (tcname);
  *tcnames = g_list_append (*tcnames, (gpointer) tcname);
  suite_add_tcase (s, tc);
  tcase_add_loop_test (tc, test_tof_elements, start, end);

  tcname = "depth";
  tc = tcase_create (tcname);
  *tcnames = g_list_append (*tcnames, (gpointer) tcname);
  suite_add_tcase (s, tc);
  tcase_add_loop_test (tc, test_tof_depth_synthetic, start, end);

  tcname = "stream";
  tc = tcase_create (tcname);
  *tcnames = g_list_append (*tcnames, (gpointer) tcname);
  suite_add_tcase (s, tc);
  tcase_set_timeout (tc, tctimeout);
  tcase_add_loop_test (tc, test_tof_stream, start, end);

  tcname = "range-switch";
  tc = tcase_create (tcname);
  *tcnames = g_list_append (*tcnames, (gpointer) tcname);
  suite_add_tcase (s, tc);
  tcase_set_timeout (tc, 40);
  tcase_add_loop_test (tc, test_tof_range_switch, start, end);

  return s;
}

void
gst_plugin_get_tof_suite (GstPluginSuite * psuite)
{
  if (psuite == NULL)
    return;

  psuite->name = "tof";
  psuite->suite = tof_suite (&psuite->tcnames, psuite->iteration, psuite->duration);
}
