/*
 * Copyright (c) 2026, arducam-tof-ventuno project.
 * SPDX-License-Identifier: BSD-3-Clause-Clear
 *
 * ToF suite: qtitofsrc and qtitofdepth (gst-plugin-tof).
 *   elements      factories, properties and pad templates
 *   stream        live qtitofsrc for the run time: DCS16 headers, consecutive groups, PTS, stats
 *   range-switch  range far -> near in PLAYING: header mode follows, settle drops counted
 *   depth         qtitofdepth on synthetic groups (known phase per pixel) vs the analytic distance,
 *                 uncalibrated (near) and with a pure-offset calibration key file (far)
 *   calibration   calibration unset: the default from GST_TOF_CALIBRATION is used; a missing default
 *                 makes READY -> PAUSED fail; the stats name the file in use
 *   flying-pixels a synthetic step 0.8 m | mixed column | 1.4 m: flying-pixel-threshold=0.05 clears exactly the
 *                 mixed column, threshold 0 changes nothing
 *   depth-z       depth-type=z with a test lens on a constant-distance group: every pixel = r / |(x, y, 1)| of the
 *                 lens within 1 mm; radial unchanged with a lens set; the calibration's [lens] group is used when
 *                 lens is unset; z without a lens, or with an unusable one, fails to start
 *   pointcloud    the "pointcloud" pad with a test lens on a constant-distance group with an invalid band: caps parse as
 *                 a FLOAT32 1x180x240x3 tensor (gst_ml_info_from_caps), a tensor meta, every valid point within 0.1 mm
 *                 of r (x, y, 1) / |(x, y, 1)|, invalid points NaN, the same with depth-type=z; no lens -> no start
 *   thread-hint   cpu-affinity=1-2 util-clamp-min=300: the pushing thread gets them (read back from the kernel and in
 *                 the stats), a second pushing thread takes them over and the first gets its own back, teardown
 *                 restores; no hint = no change; an invalid CPU list stops the start
 *   unwrap        unwrap=true on synthetic groups (left/right halves at set distances): a near group before the first
 *                 reference is unchanged, a far group (0.5 m | 2.5 m) is output as it is and becomes the reference, the
 *                 next near group reads 0.5 m | 2.5 m, a half that moved 0.4 m is invalid, a half within the tolerance
 *                 follows; unwrap=false leaves every group as computed; the stats count it
 *   references    temporal-average=2 on synthetic groups with PTS: a group flagged reference is output unaveraged and the
 *                 near window goes on across it, an unflagged far group restarts it; output-references=false: the
 *                 reference reaches no output (depth, amplitude), a GAP event at its PTS takes its place on each, unwrap
 *                 still uses it and the stats count it; by default it is output
 *   far-reference live qtitofsrc range=near far-reference-interval=10 into qtitofdepth unwrap=true: runs of 10 near
 *                 groups, each far group flagged reference + after-switch, two range writes per reference, 1-3 groups
 *                 dropped per write, the depth element counts every far group as a reference
 * The live cases (stream, range-switch, far-reference) need the camera connected and its drivers loaded; qtitofsrc runs
 * with its defaults.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE             /* sched_getaffinity, CPU_SET */
#endif

#include <math.h>
#include <sched.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <glib/gstdio.h>

#include "plugin-suite.h"
#include <gst/video/video.h>
#include <gst/ml/ml-info.h>
#include <gst/ml/gstmlmeta.h>

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
  GArray *flags;                /* guint16 */
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
    guint16 flags = GST_READ_UINT16_LE (map.data + 6);
    guint32 gseq = GST_READ_UINT32_LE (map.data + 8);
    guint64 pts = GST_BUFFER_PTS (buf);

    for (guint k = 1; k < 4; k++)
      if (GST_READ_UINT32_LE (map.data + 12 + 4 * k) !=
          GST_READ_UINT32_LE (map.data + 12 + 4 * (k - 1)) + 1)
        rec->bad_frame_seq++;
    rec->firmware = GST_READ_UINT32_LE (map.data + 42);
    g_array_append_val (rec->mode, mode);
    g_array_append_val (rec->flags, flags);
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
stream_and_record_full (const gchar * desc, guint groups, guint switch_at, TofRecord * rec,
    GstStructure ** stats, GstStructure ** depth_stats)
{
  GError *err = NULL;
  GstElement *pipeline = gst_parse_launch (desc, &err);
  GstPad *pad;

  fail_unless (pipeline != NULL, "parse: %s", err ? err->message : "?");
  memset (rec, 0, sizeof (*rec));
  rec->mode = g_array_new (FALSE, FALSE, sizeof (guint8));
  rec->flags = g_array_new (FALSE, FALSE, sizeof (guint16));
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
  if (depth_stats) {
    GstElement *d = gst_bin_get_by_name (GST_BIN (pipeline), "d");

    fail_unless (d != NULL);
    g_object_get (d, "stats", depth_stats, NULL);
    gst_object_unref (d);
  }
  gst_element_set_state (pipeline, GST_STATE_NULL);
  gst_object_unref (rec->src);
  gst_object_unref (pipeline);
}

static void
stream_and_record (const gchar * desc, guint groups, guint switch_at, TofRecord * rec,
    GstStructure ** stats)
{
  stream_and_record_full (desc, groups, switch_at, rec, stats, NULL);
}

static void
record_free (TofRecord * rec)
{
  g_array_unref (rec->mode);
  g_array_unref (rec->flags);
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

  /* explicitly uncalibrated: phi_ref 0, no correction, near mode R = 2 m */
  check_depth ("qtitofdepth calibration=\"\"", 1, 2.0, 0.0, 0.0);

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

GST_START_TEST (test_tof_calibration_default)
{
  gchar *path = NULL, *used = NULL;
  gint fd = g_file_open_tmp ("tof-cal-XXXXXX.conf", &path, NULL);
  const gchar *cal = "[calibration]\nformat=2\n"
      "[far]\nR=4.0\nphi_ref=0.75\nc0=0.02\n[near]\nR=2.0\nphi_ref=0.3\nc0=-0.01\n";
  GstElement *depth;
  GstStructure *stats;

  fail_unless (fd >= 0);
  fail_unless (g_file_set_contents (path, cal, -1, NULL));
  close (fd);

  /* unset -> the default named by GST_TOF_CALIBRATION, applied to the depth */
  g_setenv ("GST_TOF_CALIBRATION", path, TRUE);
  check_depth ("qtitofdepth", 0, 4.0, 0.75, 0.02);
  check_depth ("qtitofdepth", 1, 2.0, 0.3, -0.01);
  depth = gst_element_factory_make ("qtitofdepth", NULL);
  fail_unless_equals_int (gst_element_set_state (depth, GST_STATE_PAUSED), GST_STATE_CHANGE_SUCCESS);
  g_object_get (depth, "stats", &stats, NULL);
  used = g_strdup (gst_structure_get_string (stats, "calibration"));
  fail_unless_equals_string (used, path);
  gst_structure_free (stats);
  g_free (used);
  gst_element_set_state (depth, GST_STATE_NULL);

  /* unset and no default file -> refuses to start */
  g_setenv ("GST_TOF_CALIBRATION", "/nonexistent/tof-depth-cal.conf", TRUE);
  fail_unless_equals_int (gst_element_set_state (depth, GST_STATE_PAUSED), GST_STATE_CHANGE_FAILURE);
  gst_element_set_state (depth, GST_STATE_NULL);

  /* calibration="" -> uncalibrated, whatever the default */
  g_object_set (depth, "calibration", "", NULL);
  fail_unless_equals_int (gst_element_set_state (depth, GST_STATE_PAUSED), GST_STATE_CHANGE_SUCCESS);
  g_object_get (depth, "stats", &stats, NULL);
  fail_unless_equals_string (gst_structure_get_string (stats, "calibration"), "");
  gst_structure_free (stats);
  gst_element_set_state (depth, GST_STATE_NULL);

  gst_object_unref (depth);
  g_unsetenv ("GST_TOF_CALIBRATION");
  g_unlink (path);
  g_free (path);
}
GST_END_TEST;

/* Uncalibrated near mode (R 2 m, phi_ref 0): columns < 120 at 0.8 m, > 120 at 1.4 m, column 120 the mean of the
 * two phasors (a flying pixel, ~1.1 m). Returns the GRAY16_LE depth (mm) and the stats' flying-pixels count. */
static guint16 *
depth_of_step (gdouble threshold, guint64 * flying)
{
  GstHarness *h = gst_harness_new_parse ("qtitofdepth calibration=\"\"");
  GstBuffer *in = gst_buffer_new_allocate (NULL, TOF_SIZE, NULL), *out;
  GstStructure *stats;
  GstMapInfo map;
  gint16 *f;
  guint16 *depth = g_new (guint16, TOF_PIX);
  const gdouble amp = 200.0, phi1 = -2.0 * G_PI * 0.8 / 2.0, phi2 = -2.0 * G_PI * 1.4 / 2.0;
  GstElement *depth_el = gst_harness_find_element (h, "qtitofdepth");

  fail_unless (depth_el != NULL);
  g_object_set (depth_el, "flying-pixel-threshold", threshold, NULL);
  gst_buffer_map (in, &map, GST_MAP_WRITE);
  memset (map.data, 0, TOF_HDR);
  GST_WRITE_UINT32_LE (map.data, TOF_MAGIC);
  GST_WRITE_UINT16_LE (map.data + 4, TOF_HDR);
  map.data[40] = map.data[41] = 1;
  f = (gint16 *) (map.data + TOF_HDR);
  for (guint p = 0; p < TOF_PIX; p++) {
    guint x = p % TOF_W;
    gdouble c, s;

    if (x == TOF_W / 2) {
      c = 0.5 * amp * (cos (phi1) + cos (phi2));
      s = 0.5 * amp * (sin (phi1) + sin (phi2));
    } else {
      c = amp * cos (x < TOF_W / 2 ? phi1 : phi2);
      s = amp * sin (x < TOF_W / 2 ? phi1 : phi2);
    }
    f[p] = GINT16_TO_LE ((gint16) lround (c));
    f[TOF_PIX + p] = GINT16_TO_LE ((gint16) lround (s));
    f[2 * TOF_PIX + p] = GINT16_TO_LE ((gint16) - lround (c));
    f[3 * TOF_PIX + p] = GINT16_TO_LE ((gint16) - lround (s));
  }
  gst_buffer_unmap (in, &map);

  gst_harness_set_src_caps_str (h, TOF_DCS_CAPS);
  gst_harness_set_sink_caps_str (h,
      "video/x-raw, format=(string)GRAY16_LE, width=(int)240, height=(int)180, framerate=(fraction)30/1");
  fail_unless_equals_int (gst_harness_push (h, in), GST_FLOW_OK);
  out = gst_harness_pull (h);
  fail_unless (out != NULL);
  gst_buffer_map (out, &map, GST_MAP_READ);
  fail_unless (map.size >= TOF_PIX * 2);
  for (guint p = 0; p < TOF_PIX; p++)
    depth[p] = GST_READ_UINT16_LE (map.data + 2 * p);
  gst_buffer_unmap (out, &map);
  gst_buffer_unref (out);
  g_object_get (depth_el, "stats", &stats, NULL);
  fail_unless (gst_structure_get_uint64 (stats, "flying-pixels", flying));
  gst_structure_free (stats);
  gst_object_unref (depth_el);
  gst_harness_teardown (h);
  return depth;
}

GST_START_TEST (test_tof_flying_pixels)
{
  guint64 n_off = 0, n_on = 0;
  guint16 *off = depth_of_step (0.0, &n_off), *on = depth_of_step (0.05, &n_on);

  /* the step itself: 800 | ~1100 | 1400 mm without the filter */
  fail_unless (ABS (off[90 * TOF_W + 60] - 800) <= 1, "left %u", off[90 * TOF_W + 60]);
  fail_unless (ABS (off[90 * TOF_W + 180] - 1400) <= 1, "right %u", off[90 * TOF_W + 180]);
  fail_unless (off[90 * TOF_W + 120] > 900 && off[90 * TOF_W + 120] < 1300, "mixed %u",
      off[90 * TOF_W + 120]);
  fail_unless_equals_uint64 (n_off, 0);
  /* with the filter: the mixed column (inside the 1-pixel border) is invalid, nothing else changes */
  for (guint p = 0; p < TOF_PIX; p++) {
    guint x = p % TOF_W, y = p / TOF_W;

    if (x == TOF_W / 2 && y > 0 && y < TOF_H - 1)
      fail_unless (on[p] == 0, "mixed pixel %u,%u kept (%u)", x, y, on[p]);
    else
      fail_unless (on[p] == off[p], "pixel %u,%u changed %u -> %u", x, y, off[p], on[p]);
  }
  fail_unless_equals_uint64 (n_on, TOF_H - 2);
  g_free (off);
  g_free (on);
}
GST_END_TEST;

/* GRAY16_LE depth (mm) of one uncalibrated near-mode group whose pixels all read d metres (phi_ref 0, R 2 m) */
static guint16 *
depth_of_constant (const gchar * props, gdouble d)
{
  gchar *launch = g_strdup_printf ("qtitofdepth calibration=\"\" %s", props);
  GstHarness *h = gst_harness_new_parse (launch);
  GstBuffer *in = gst_buffer_new_allocate (NULL, TOF_SIZE, NULL), *out;
  GstMapInfo map;
  gint16 *f;
  guint16 *depth = g_new (guint16, TOF_PIX);
  const gdouble phi = -2.0 * G_PI * d / 2.0;
  const gint16 c = (gint16) lround (200.0 * cos (phi)), s = (gint16) lround (200.0 * sin (phi));

  gst_buffer_map (in, &map, GST_MAP_WRITE);
  memset (map.data, 0, TOF_HDR);
  GST_WRITE_UINT32_LE (map.data, TOF_MAGIC);
  GST_WRITE_UINT16_LE (map.data + 4, TOF_HDR);
  map.data[40] = map.data[41] = 1;
  f = (gint16 *) (map.data + TOF_HDR);
  for (guint p = 0; p < TOF_PIX; p++) {
    f[p] = GINT16_TO_LE (c);
    f[TOF_PIX + p] = GINT16_TO_LE (s);
    f[2 * TOF_PIX + p] = GINT16_TO_LE ((gint16) - c);
    f[3 * TOF_PIX + p] = GINT16_TO_LE ((gint16) - s);
  }
  gst_buffer_unmap (in, &map);
  gst_harness_set_src_caps_str (h, TOF_DCS_CAPS);
  gst_harness_set_sink_caps_str (h,
      "video/x-raw, format=(string)GRAY16_LE, width=(int)240, height=(int)180, framerate=(fraction)30/1");
  fail_unless_equals_int (gst_harness_push (h, in), GST_FLOW_OK);
  out = gst_harness_pull (h);
  fail_unless (out != NULL);
  gst_buffer_map (out, &map, GST_MAP_READ);
  fail_unless (map.size >= TOF_PIX * 2);
  for (guint p = 0; p < TOF_PIX; p++)
    depth[p] = GST_READ_UINT16_LE (map.data + 2 * p);
  gst_buffer_unmap (out, &map);
  gst_buffer_unref (out);
  gst_harness_teardown (h);
  g_free (launch);
  return depth;
}

/* z / r of pixel (u, v): 1 / |(x, y, 1)|, x = xd (1 + k1 rd^2 + k2 rd^4), xd = (u - cx) / f (the lens model of qtitofdepth) */
static gdouble
lens_zscale (guint u, guint v, gdouble f, gdouble cx, gdouble cy, gdouble k1, gdouble k2)
{
  gdouble xd = (u - cx) / f, yd = (v - cy) / f, r2 = xd * xd + yd * yd, k = 1.0 + k1 * r2 + k2 * r2 * r2;

  return 1.0 / sqrt (1.0 + xd * k * xd * k + yd * k * yd * k);
}

static gchar *
write_tmp (const gchar * contents)
{
  gchar *path = NULL;
  gint fd = g_file_open_tmp ("tof-lens-XXXXXX.conf", &path, NULL);

  fail_unless (fd >= 0);
  fail_unless (g_file_set_contents (path, contents, -1, NULL));
  close (fd);
  return path;
}

/* READY -> PAUSED of a qtitofdepth with these properties: success or failure */
static GstStateChangeReturn
start_depth (const gchar * calibration, const gchar * lens, gint depth_type, gchar ** lens_used)
{
  GstElement *e = gst_element_factory_make ("qtitofdepth", NULL);
  GstStateChangeReturn r;

  g_object_set (e, "calibration", calibration, "depth-type", depth_type, NULL);
  if (lens)
    g_object_set (e, "lens", lens, NULL);
  r = gst_element_set_state (e, GST_STATE_PAUSED);
  if (r == GST_STATE_CHANGE_SUCCESS && lens_used) {
    GstStructure *stats;

    g_object_get (e, "stats", &stats, NULL);
    *lens_used = g_strdup (gst_structure_get_string (stats, "lens"));
    fail_unless_equals_string (gst_structure_get_string (stats, "depth-type"), depth_type ? "z" : "radial");
    gst_structure_free (stats);
  }
  gst_element_set_state (e, GST_STATE_NULL);
  gst_object_unref (e);
  return r;
}

GST_START_TEST (test_tof_depth_z)
{
  const gdouble d = 1.2, f = 200.0, cx = 119.5, cy = 89.5, k1 = 0.1;
  gchar *lens = write_tmp ("[lens]\nf=200\ncx=119.5\ncy=89.5\nk1=0.1\nk2=0\nwidth=240\nheight=180\n");
  gchar *bad_f = write_tmp ("[lens]\nf=0\ncx=119.5\ncy=89.5\nk1=0\n");
  gchar *no_group = write_tmp ("[calibration]\nformat=2\n");
  gchar *cal = write_tmp ("[calibration]\nformat=2\n[far]\nR=4.0\nphi_ref=0\nc0=0\n[near]\nR=2.0\nphi_ref=0\n"
      "c0=0\n[lens]\nf=200\ncx=119.5\ncy=89.5\nk1=0.1\n");
  gchar *props, *used = NULL;
  guint16 *radial = depth_of_constant ("", d), *radial_lens, *z;
  gint worst = 0;
  guint worst_px = 0;

  props = g_strdup_printf ("lens=%s", lens);
  radial_lens = depth_of_constant (props, d);
  g_free (props);
  props = g_strdup_printf ("depth-type=z lens=%s", lens);
  z = depth_of_constant (props, d);
  g_free (props);
  for (guint p = 0; p < TOF_PIX; p++) {
    /* radial mm (rounded) times z/r vs z from the unrounded distance: within 1 mm */
    gint e = z[p] - (gint) lround (radial[p] * lens_zscale (p % TOF_W, p / TOF_W, f, cx, cy, k1, 0.0));

    fail_unless (ABS (radial[p] - 1200) <= 1, "radial %u at pixel %u", radial[p], p);
    fail_unless (radial_lens[p] == radial[p], "radial changed by a lens at pixel %u", p);
    if (ABS (e) > ABS (worst)) {
      worst = e;
      worst_px = p;
    }
  }
  fail_unless (ABS (worst) <= 1, "z: max error %d mm at pixel %u", worst, worst_px);
  /* the centre keeps r, the corner (0, 0) is ~0.785 r with this lens */
  fail_unless (ABS (z[90 * TOF_W + 120] - 1200) <= 1, "centre z %u", z[90 * TOF_W + 120]);
  fail_unless (z[0] > 930 && z[0] < 955, "corner z %u", z[0]);

  /* lens unset: the calibration's [lens] group */
  fail_unless_equals_int (start_depth (cal, NULL, 1, &used), GST_STATE_CHANGE_SUCCESS);
  fail_unless_equals_string (used, cal);
  g_free (used);
  /* radial without any lens: fine, stats lens "" */
  fail_unless_equals_int (start_depth ("", NULL, 0, &used), GST_STATE_CHANGE_SUCCESS);
  fail_unless_equals_string (used, "");
  g_free (used);
  /* z without a lens, or with an unusable one, does not start; an explicit unusable lens fails also for radial */
  fail_unless_equals_int (start_depth ("", NULL, 1, NULL), GST_STATE_CHANGE_FAILURE);
  fail_unless_equals_int (start_depth ("", bad_f, 1, NULL), GST_STATE_CHANGE_FAILURE);
  fail_unless_equals_int (start_depth ("", no_group, 1, NULL), GST_STATE_CHANGE_FAILURE);
  fail_unless_equals_int (start_depth ("", "/nonexistent/tof-lens.conf", 0, NULL), GST_STATE_CHANGE_FAILURE);

  g_free (radial);
  g_free (radial_lens);
  g_free (z);
  g_unlink (lens);
  g_unlink (bad_f);
  g_unlink (no_group);
  g_unlink (cal);
  g_free (lens);
  g_free (bad_f);
  g_free (no_group);
  g_free (cal);
}
GST_END_TEST;

/* point cloud (X, Y, Z per pixel, m) of one uncalibrated near-mode group: every pixel at d metres, rows < 10 invalid
 * (I = Q = 0); checks caps and meta; returns the floats and the element's radial distance of the valid pixels */
static gfloat *
pointcloud_of_constant (const gchar * props, gdouble d, gdouble * r_out)
{
  gchar *launch = g_strdup_printf ("qtitofdepth calibration=\"\" %s", props);
  GstElement *e = gst_parse_launch (launch, NULL);
  GstHarness *h;
  GstBuffer *in = gst_buffer_new_allocate (NULL, TOF_SIZE, NULL), *out;
  GstMapInfo map;
  GstMLInfo info;
  GstMLTensorMeta *meta;
  GstCaps *caps;
  gint16 *f;
  gfloat *pts = g_new (gfloat, TOF_PIX * 3);
  const gdouble phi = -2.0 * G_PI * d / 2.0;
  const gint16 c = (gint16) lround (200.0 * cos (phi)), s = (gint16) lround (200.0 * sin (phi));
  gdouble u = -atan2 (2.0 * s, 2.0 * c);

  fail_unless (e != NULL);
  h = gst_harness_new_with_element (e, "sink", "pointcloud");
  gst_buffer_map (in, &map, GST_MAP_WRITE);
  memset (map.data, 0, TOF_HDR);
  GST_WRITE_UINT32_LE (map.data, TOF_MAGIC);
  GST_WRITE_UINT16_LE (map.data + 4, TOF_HDR);
  map.data[40] = map.data[41] = 1;
  f = (gint16 *) (map.data + TOF_HDR);
  for (guint p = 0; p < TOF_PIX; p++) {
    gboolean bad = p / TOF_W < 10;

    f[p] = GINT16_TO_LE (bad ? 0 : c);
    f[TOF_PIX + p] = GINT16_TO_LE (bad ? 0 : s);
    f[2 * TOF_PIX + p] = GINT16_TO_LE (bad ? 0 : (gint16) - c);
    f[3 * TOF_PIX + p] = GINT16_TO_LE (bad ? 0 : (gint16) - s);
  }
  gst_buffer_unmap (in, &map);
  gst_harness_set_src_caps_str (h, TOF_DCS_CAPS);
  fail_unless_equals_int (gst_harness_push (h, in), GST_FLOW_OK);
  out = gst_harness_pull (h);
  fail_unless (out != NULL);

  caps = gst_pad_get_current_caps (h->sinkpad);
  fail_unless (caps != NULL);
  fail_unless (gst_ml_info_from_caps (&info, caps), "caps %" GST_PTR_FORMAT, caps);
  fail_unless (GST_ML_INFO_TYPE (&info) == GST_ML_TYPE_FLOAT32 && GST_ML_INFO_N_TENSORS (&info) == 1 &&
      GST_ML_INFO_N_DIMENSIONS (&info, 0) == 4 && GST_ML_INFO_TENSOR_DIM (&info, 0, 0) == 1 &&
      GST_ML_INFO_TENSOR_DIM (&info, 0, 1) == TOF_H && GST_ML_INFO_TENSOR_DIM (&info, 0, 2) == TOF_W &&
      GST_ML_INFO_TENSOR_DIM (&info, 0, 3) == 3, "caps %" GST_PTR_FORMAT, caps);
  fail_unless_equals_uint64 (gst_ml_info_size (&info), TOF_PIX * 3 * sizeof (gfloat));
  gst_caps_unref (caps);
  meta = gst_buffer_get_ml_tensor_meta (out);
  fail_unless (meta != NULL);
  fail_unless (meta->type == GST_ML_TYPE_FLOAT32 && meta->n_dimensions == 4 && meta->dimensions[1] == TOF_H &&
      meta->dimensions[2] == TOF_W && meta->dimensions[3] == 3);
  fail_unless_equals_string (g_quark_to_string (meta->name), "pointcloud");

  gst_buffer_map (out, &map, GST_MAP_READ);
  fail_unless_equals_uint64 (map.size, TOF_PIX * 3 * sizeof (gfloat));
  memcpy (pts, map.data, map.size);
  gst_buffer_unmap (out, &map);
  gst_buffer_unref (out);
  gst_harness_teardown (h);
  g_free (launch);
  u -= 2.0 * G_PI * floor (u / (2.0 * G_PI));
  *r_out = 2.0 * u / (2.0 * G_PI);      /* R = 2 m, phi_ref 0: the distance of the rounded I/Q */
  return pts;
}

GST_START_TEST (test_tof_pointcloud)
{
  const gdouble f = 200.0, cx = 119.5, cy = 89.5, k1 = 0.1;
  gchar *lens = write_tmp ("[lens]\nf=200\ncx=119.5\ncy=89.5\nk1=0.1\n");
  gchar *props;
  gfloat *pts[2];
  gdouble r = 0.0, worst = 0.0;
  GstElement *e;

  props = g_strdup_printf ("lens=%s", lens);
  pts[0] = pointcloud_of_constant (props, 1.2, &r);
  g_free (props);
  props = g_strdup_printf ("depth-type=z lens=%s", lens);
  pts[1] = pointcloud_of_constant (props, 1.2, &r);
  g_free (props);
  fail_unless (fabs (r - 1.2) < 0.002, "r %f", r);
  for (guint k = 0; k < 2; k++) {
    for (guint p = 0; p < TOF_PIX; p++) {
      guint uu = p % TOF_W, vv = p / TOF_W;
      gdouble xd = (uu - cx) / f, yd = (vv - cy) / f, r2 = xd * xd + yd * yd, s = 1.0 + k1 * r2;
      gdouble x = xd * s, y = yd * s, n = sqrt (1.0 + x * x + y * y);
      const gfloat *q = pts[k] + 3 * p;

      if (vv < 10) {
        fail_unless (isnan (q[0]) && isnan (q[1]) && isnan (q[2]), "invalid pixel %u,%u not NaN", uu, vv);
        continue;
      }
      worst = MAX (worst, fabs (q[0] - r * x / n));
      worst = MAX (worst, fabs (q[1] - r * y / n));
      worst = MAX (worst, fabs (q[2] - r / n));
    }
  }
  fail_unless (worst < 1e-4, "max point error %.6f m", worst);
  /* corner (0, 10) is up-left: x < 0, y < 0 */
  fail_unless (pts[0][3 * (10 * TOF_W)] < 0.0f && pts[0][3 * (10 * TOF_W) + 1] < 0.0f);

  /* no lens: the pad cannot stream */
  e = gst_element_factory_make ("qtitofdepth", NULL);
  g_object_set (e, "calibration", "", NULL);
  fail_unless (gst_element_request_pad_simple (e, "pointcloud") != NULL);
  fail_unless_equals_int (gst_element_set_state (e, GST_STATE_PAUSED), GST_STATE_CHANGE_FAILURE);
  gst_element_set_state (e, GST_STATE_NULL);
  gst_object_unref (e);

  g_free (pts[0]);
  g_free (pts[1]);
  g_unlink (lens);
  g_free (lens);
}
GST_END_TEST;

/* Near-mode group whose pixels all read 0.5 m (phi_ref 0, R 2 m): columns < 80 amplitude 200, 80..159 amplitude 10,
 * >= 160 amplitude 5. Returns the GRAY16_LE depth (mm), the stats' wrap-flagged count and the push result. */
static guint16 *
depth_of_bands (const gchar * calibration, gdouble rho, guint64 * wrapped, GstFlowReturn * flow)
{
  gchar *launch = g_strdup_printf ("qtitofdepth calibration=\"%s\" wrap-reflectivity=%g", calibration, rho);
  GstHarness *h = gst_harness_new_parse (launch);
  GstElement *el = gst_harness_find_element (h, "qtitofdepth");
  GstBuffer *in = gst_buffer_new_allocate (NULL, TOF_SIZE, NULL), *out;
  GstStructure *stats;
  GstMapInfo map;
  gint16 *f;
  guint16 *depth = g_new0 (guint16, TOF_PIX);

  gst_buffer_map (in, &map, GST_MAP_WRITE);
  memset (map.data, 0, TOF_HDR);
  GST_WRITE_UINT32_LE (map.data, TOF_MAGIC);
  GST_WRITE_UINT16_LE (map.data + 4, TOF_HDR);
  map.data[40] = map.data[41] = 1;
  f = (gint16 *) (map.data + TOF_HDR);
  for (guint p = 0; p < TOF_PIX; p++) {
    guint x = p % TOF_W;
    gint16 a = x < 80 ? 200 : x < 160 ? 10 : 5;

    /* phi = -2 pi 0.5 / 2 = -pi / 2: F0 = F2 = 0, F1 = -a, F3 = a, so I = 0, Q = -2a and the amplitude is a */
    f[TOF_PIX + p] = GINT16_TO_LE (-a);
    f[3 * TOF_PIX + p] = GINT16_TO_LE (a);
  }
  gst_buffer_unmap (in, &map);
  gst_harness_set_src_caps_str (h, TOF_DCS_CAPS);
  gst_harness_set_sink_caps_str (h,
      "video/x-raw, format=(string)GRAY16_LE, width=(int)240, height=(int)180, framerate=(fraction)30/1");
  *flow = gst_harness_push (h, in);
  if (*flow == GST_FLOW_OK) {
    out = gst_harness_pull (h);
    fail_unless (out != NULL);
    gst_buffer_map (out, &map, GST_MAP_READ);
    fail_unless (map.size >= TOF_PIX * 2);
    for (guint p = 0; p < TOF_PIX; p++)
      depth[p] = GST_READ_UINT16_LE (map.data + 2 * p);
    gst_buffer_unmap (out, &map);
    gst_buffer_unref (out);
  }
  g_object_get (el, "stats", &stats, NULL);
  fail_unless (gst_structure_get_uint64 (stats, "wrap-flagged", wrapped));
  gst_structure_free (stats);
  gst_object_unref (el);
  gst_harness_teardown (h);
  g_free (launch);
  return depth;
}

GST_START_TEST (test_tof_wrap)
{
  /* bound at r = 0.5 m, R = 2 m, rho 2: 2 * 20 / 2.5^2 + 3 * 0.5 = 7.9 counts -> amplitude 5 may be wrapped, 10 not */
  gchar *cal = write_tmp ("[calibration]\nformat=2\n"
      "[far]\nR=4.0\nphi_ref=0\nc0=0\nwrap_k=20\nwrap_sigma=0.5\n"
      "[near]\nR=2.0\nphi_ref=0\nc0=0\nwrap_k=20\nwrap_sigma=0.5\n");
  guint64 n_off = 1, n_on = 0, n_err = 0;
  GstFlowReturn flow;
  guint16 *off = depth_of_bands (cal, 0.0, &n_off, &flow), *on, *err;

  fail_unless_equals_int (flow, GST_FLOW_OK);
  fail_unless_equals_uint64 (n_off, 0);
  for (guint p = 0; p < TOF_PIX; p++)
    fail_unless (ABS (off[p] - 500) <= 1, "pixel %u: %u mm without the test", p, off[p]);
  on = depth_of_bands (cal, 2.0, &n_on, &flow);
  fail_unless_equals_int (flow, GST_FLOW_OK);
  for (guint p = 0; p < TOF_PIX; p++) {
    if (p % TOF_W >= 160)
      fail_unless (on[p] == 0, "weak pixel %u kept (%u)", p, on[p]);
    else
      fail_unless (on[p] == off[p], "pixel %u changed %u -> %u", p, off[p], on[p]);
  }
  fail_unless_equals_uint64 (n_on, 80 * TOF_H);
  /* enabled without the calibration's wrap reference: an error, no output */
  err = depth_of_bands ("", 2.0, &n_err, &flow);
  fail_unless_equals_int (flow, GST_FLOW_ERROR);
  fail_unless_equals_uint64 (n_err, 0);
  g_free (off);
  g_free (on);
  g_free (err);
  g_unlink (cal);
  g_free (cal);
}
GST_END_TEST;

/* uclamp.min of the calling thread (sched_getattr; struct sched_attr, SCHED_ATTR_SIZE_VER1) */
static gint
own_util_min (void)
{
  struct
  {
    guint32 size, policy;
    guint64 flags;
    gint32 nice;
    guint32 priority;
    guint64 runtime, deadline, period;
    guint32 util_min, util_max;
  } a = { 0, };

  fail_unless (syscall (SYS_sched_getattr, 0, &a, sizeof (a), 0) == 0);
  return (gint) a.util_min;
}

typedef struct
{
  GstHarness *h;
  cpu_set_t cpus;
  gint util;
  GstFlowReturn flow;
} HintPush;

static gpointer
hint_push_thread (gpointer data)
{
  HintPush *hp = data;

  hp->flow = gst_harness_push (hp->h, synthetic_group (1, 200.0));
  sched_getaffinity (0, sizeof (cpu_set_t), &hp->cpus);
  hp->util = own_util_min ();
  return NULL;
}

static void
check_hint_stats (GstElement * el, const gchar * cpus, gint util)
{
  GstStructure *stats;
  gint got = -2;

  g_object_get (el, "stats", &stats, NULL);
  fail_unless_equals_string (gst_structure_get_string (stats, "thread-cpus"), cpus);
  fail_unless (gst_structure_get_int (stats, "thread-util-clamp-min", &got));
  fail_unless_equals_int (got, util);
  gst_structure_free (stats);
}

GST_START_TEST (test_tof_thread_hint)
{
  const gchar *bad[] = { "0-99", "abc", "3-1", "1,", "-1", "1-", " " };
  cpu_set_t before, now, want;
  gint util_before = own_util_min ();
  GstHarness *h;
  GstElement *el;
  GThread *t;
  HintPush hp = { 0, };

  fail_unless (sched_getaffinity (0, sizeof (before), &before) == 0);
  fail_unless (util_before != 300);
  CPU_ZERO (&want);
  CPU_SET (1, &want);
  CPU_SET (2, &want);
  fail_if (CPU_EQUAL (&before, &want), "the test thread is already on CPUs 1-2");

  /* the hint lands on the thread that pushes (here this one) and shows in the stats */
  h = gst_harness_new_parse ("qtitofdepth calibration=\"\" cpu-affinity=1-2 util-clamp-min=300");
  el = gst_harness_find_element (h, "qtitofdepth");
  gst_harness_set_src_caps_str (h, TOF_DCS_CAPS);
  gst_harness_set_sink_caps_str (h,
      "video/x-raw, format=(string)GRAY16_LE, width=(int)240, height=(int)180, framerate=(fraction)30/1");
  check_hint_stats (el, "", -1);
  fail_unless_equals_int (gst_harness_push (h, synthetic_group (1, 200.0)), GST_FLOW_OK);
  gst_buffer_unref (gst_harness_pull (h));
  fail_unless (sched_getaffinity (0, sizeof (now), &now) == 0);
  fail_unless (CPU_EQUAL (&now, &want), "affinity not applied");
  fail_unless_equals_int (own_util_min (), 300);
  check_hint_stats (el, "1-2", 300);

  /* another thread takes over: it gets the hint, this one its own values back */
  hp.h = h;
  t = g_thread_new ("hint-push", hint_push_thread, &hp);
  g_thread_join (t);
  fail_unless_equals_int (hp.flow, GST_FLOW_OK);
  gst_buffer_unref (gst_harness_pull (h));
  fail_unless (CPU_EQUAL (&hp.cpus, &want), "affinity not applied to the second thread");
  fail_unless_equals_int (hp.util, 300);
  fail_unless (sched_getaffinity (0, sizeof (now), &now) == 0);
  fail_unless (CPU_EQUAL (&now, &before), "affinity not restored after the takeover");
  fail_unless_equals_int (own_util_min (), util_before);

  /* back on this thread, then stop: restored */
  fail_unless_equals_int (gst_harness_push (h, synthetic_group (1, 200.0)), GST_FLOW_OK);
  gst_buffer_unref (gst_harness_pull (h));
  fail_unless_equals_int (own_util_min (), 300);
  gst_object_unref (el);
  gst_harness_teardown (h);
  fail_unless (sched_getaffinity (0, sizeof (now), &now) == 0);
  fail_unless (CPU_EQUAL (&now, &before), "affinity not restored at stop");
  fail_unless_equals_int (own_util_min (), util_before);

  /* no hint: the thread is left alone */
  h = gst_harness_new_parse ("qtitofdepth calibration=\"\"");
  el = gst_harness_find_element (h, "qtitofdepth");
  gst_harness_set_src_caps_str (h, TOF_DCS_CAPS);
  gst_harness_set_sink_caps_str (h,
      "video/x-raw, format=(string)GRAY16_LE, width=(int)240, height=(int)180, framerate=(fraction)30/1");
  fail_unless_equals_int (gst_harness_push (h, synthetic_group (1, 200.0)), GST_FLOW_OK);
  gst_buffer_unref (gst_harness_pull (h));
  fail_unless (sched_getaffinity (0, sizeof (now), &now) == 0);
  fail_unless (CPU_EQUAL (&now, &before));
  fail_unless_equals_int (own_util_min (), util_before);
  check_hint_stats (el, "", -1);
  gst_object_unref (el);
  gst_harness_teardown (h);

  /* an invalid CPU list stops the start */
  for (guint i = 0; i < G_N_ELEMENTS (bad); i++) {
    el = gst_element_factory_make ("qtitofdepth", NULL);
    g_object_set (el, "calibration", "", "cpu-affinity", bad[i], NULL);
    fail_unless (gst_element_set_state (el, GST_STATE_PAUSED) == GST_STATE_CHANGE_FAILURE,
        "cpu-affinity \"%s\" accepted", bad[i]);
    gst_element_set_state (el, GST_STATE_NULL);
    gst_object_unref (el);
  }
}
GST_END_TEST;

/* A group whose left half (x < 120) reads dl and right half dr (phi_ref 0, c0 0: phi = -2 pi d / R), amplitude 200 */
static GstBuffer *
group_at (guint8 mode, gdouble dl, gdouble dr)
{
  GstBuffer *buf = gst_buffer_new_allocate (NULL, TOF_SIZE, NULL);
  gdouble r = mode ? 2.0 : 4.0;
  GstMapInfo map;
  gint16 *f;

  gst_buffer_map (buf, &map, GST_MAP_WRITE);
  memset (map.data, 0, TOF_HDR);
  GST_WRITE_UINT32_LE (map.data, TOF_MAGIC);
  GST_WRITE_UINT16_LE (map.data + 4, TOF_HDR);
  map.data[40] = map.data[41] = mode;
  f = (gint16 *) (map.data + TOF_HDR);
  for (guint p = 0; p < TOF_PIX; p++) {
    gdouble phi = -2.0 * G_PI * (p % TOF_W < TOF_W / 2 ? dl : dr) / r;
    gint16 c = (gint16) lround (200.0 * cos (phi)), s = (gint16) lround (200.0 * sin (phi));

    f[p] = GINT16_TO_LE (c);
    f[TOF_PIX + p] = GINT16_TO_LE (s);
    f[2 * TOF_PIX + p] = GINT16_TO_LE (-c);
    f[3 * TOF_PIX + p] = GINT16_TO_LE (-s);
  }
  gst_buffer_unmap (buf, &map);
  return buf;
}

/* push one group, check every pixel of each half of the GRAY16_LE depth against the expected mm (0 = invalid) */
static void
push_expect (GstHarness * h, GstBuffer * in, gint want_l, gint want_r, const gchar * what)
{
  GstBuffer *out;
  GstMapInfo map;

  fail_unless_equals_int (gst_harness_push (h, in), GST_FLOW_OK);
  out = gst_harness_pull (h);
  fail_unless (out != NULL, "%s: no output", what);
  gst_buffer_map (out, &map, GST_MAP_READ);
  fail_unless (map.size >= TOF_PIX * 2);
  for (guint p = 0; p < TOF_PIX; p++) {
    gint got = GST_READ_UINT16_LE (map.data + 2 * p), want = p % TOF_W < TOF_W / 2 ? want_l : want_r;

    fail_unless (want == 0 ? got == 0 : ABS (got - want) <= 3, "%s: pixel %u %d mm, expected %d", what, p, got, want);
  }
  gst_buffer_unmap (out, &map);
  gst_buffer_unref (out);
}

static GstHarness *
unwrap_harness (const gchar * cal, gboolean unwrap)
{
  gchar *launch = g_strdup_printf ("qtitofdepth calibration=\"%s\" unwrap=%s", cal, unwrap ? "true" : "false");
  GstHarness *h = gst_harness_new_parse (launch);

  g_free (launch);
  gst_harness_set_src_caps_str (h, TOF_DCS_CAPS);
  gst_harness_set_sink_caps_str (h,
      "video/x-raw, format=(string)GRAY16_LE, width=(int)240, height=(int)180, framerate=(fraction)30/1");
  return h;
}

GST_START_TEST (test_tof_unwrap)
{
  gchar *cal = write_tmp ("[calibration]\nformat=2\n"
      "[far]\nR=4.0\nphi_ref=0\nc0=0\n[near]\nR=2.0\nphi_ref=0\nc0=0\n");
  GstHarness *h = unwrap_harness (cal, TRUE);
  GstElement *el = gst_harness_find_element (h, "qtitofdepth");
  GstStructure *stats;
  guint64 refs = 0, unw = 0, und = 0;
  const guint64 half = TOF_PIX / 2;

  push_expect (h, group_at (1, 0.5, 0.5), 500, 500, "near before the first reference");
  push_expect (h, group_at (0, 0.5, 2.5), 500, 2500, "far reference");
  push_expect (h, group_at (1, 0.5, 0.5), 500, 2500, "near after the reference");
  push_expect (h, group_at (1, 0.5, 0.9), 500, 0, "right half moved 0.4 m");
  push_expect (h, group_at (1, 0.55, 0.5), 550, 2500, "left half within the tolerance, right back");
  g_object_get (el, "stats", &stats, NULL);
  fail_unless (gst_structure_get_uint64 (stats, "unwrap-references", &refs));
  fail_unless (gst_structure_get_uint64 (stats, "unwrapped", &unw));
  fail_unless (gst_structure_get_uint64 (stats, "unwrap-undecided", &und));
  fail_unless_equals_uint64 (refs, 1);
  fail_unless_equals_uint64 (unw, 2 * half);
  fail_unless_equals_uint64 (und, half);
  gst_structure_free (stats);
  gst_object_unref (el);
  gst_harness_teardown (h);

  /* off: every group as computed */
  h = unwrap_harness (cal, FALSE);
  push_expect (h, group_at (1, 0.5, 0.5), 500, 500, "off: near");
  push_expect (h, group_at (0, 0.5, 2.5), 500, 2500, "off: far");
  push_expect (h, group_at (1, 0.5, 0.5), 500, 500, "off: near after a far group");
  push_expect (h, group_at (1, 0.5, 0.9), 500, 900, "off: moved half");
  gst_harness_teardown (h);
  g_unlink (cal);
  g_free (cal);
}
GST_END_TEST;

GST_START_TEST (test_tof_far_reference)
{
  TofRecord rec;
  GstStructure *stats = NULL, *dstats = NULL;
  guint64 refs = 0, switches = 0, dropped = 0, drefs = 0;
  guint n_far = 0, run = 0, runs_bad = 0, flags_bad = 0, first_far = G_MAXUINT;

  stream_and_record_full ("qtitofsrc name=s range=near far-reference-interval=10 num-buffers=166 ! "
      "qtitofdepth name=d unwrap=true ! fakesink sync=false", 166, 0, &rec, &stats, &dstats);
  fail_unless_equals_int (rec.bad_header, 0);
  for (guint i = 0; i < rec.mode->len; i++) {
    guint8 m = g_array_index (rec.mode, guint8, i);
    guint16 fl = g_array_index (rec.flags, guint16, i);

    if (m == 0) {
      /* (1 << 1) after-switch, (1 << 2) reference */
      if ((fl & 0x6) != 0x6)
        flags_bad++;
      if (run != 10)
        runs_bad++;
      if (first_far == G_MAXUINT)
        first_far = i;
      n_far++;
      run = 0;
    } else {
      if (fl & 0x4)
        flags_bad++;
      run++;
    }
  }
  fail_unless (first_far == 10, "first far group at %u", first_far);
  fail_unless_equals_int (runs_bad, 0);
  fail_unless_equals_int (flags_bad, 0);
  fail_unless_equals_int (n_far, 15);
  fail_unless (gst_structure_get_uint64 (stats, "far-references", &refs));
  fail_unless (gst_structure_get_uint64 (stats, "range-switches", &switches));
  fail_unless (gst_structure_get_uint64 (stats, "groups-dropped-switch", &dropped));
  fail_unless (gst_structure_get_uint64 (dstats, "unwrap-references", &drefs));
  fail_unless_equals_uint64 (refs, n_far);
  fail_unless_equals_uint64 (switches, 2 * (guint64) n_far);
  fail_unless (dropped >= switches && dropped <= 3 * switches, "dropped %" G_GUINT64_FORMAT " for %"
      G_GUINT64_FORMAT " writes", dropped, switches);
  fail_unless_equals_uint64 (drefs, n_far);
  gst_structure_free (stats);
  gst_structure_free (dstats);
  record_free (&rec);
}
GST_END_TEST;

/* references: buffers and GAP events seen on an output */
typedef struct
{
  guint buffers;
  guint gaps;
  GstClockTime gap_ts;
} RefCount;

static GstPadProbeReturn
ref_count_probe (GstPad * pad, GstPadProbeInfo * info, gpointer data)
{
  RefCount *c = data;

  if (GST_PAD_PROBE_INFO_TYPE (info) & GST_PAD_PROBE_TYPE_BUFFER) {
    c->buffers++;
  } else if (GST_EVENT_TYPE (GST_PAD_PROBE_INFO_EVENT (info)) == GST_EVENT_GAP) {
    gst_event_parse_gap (GST_PAD_PROBE_INFO_EVENT (info), &c->gap_ts, NULL);
    c->gaps++;
  }
  return GST_PAD_PROBE_OK;
}

/* group n of a 30/s stream; reference = the header flag qtitofsrc far-reference-interval sets */
static GstBuffer *
timed (GstBuffer * buf, guint n, gboolean reference)
{
  GST_BUFFER_PTS (buf) = n * GST_SECOND / 30;
  GST_BUFFER_DURATION (buf) = GST_SECOND / 30;
  if (reference) {
    GstMapInfo map;

    gst_buffer_map (buf, &map, GST_MAP_WRITE);
    GST_WRITE_UINT16_LE (map.data + 6, 1 << 2);
    gst_buffer_unmap (buf, &map);
  }
  return buf;
}

static GstHarness *
props_harness (const gchar * cal, const gchar * props)
{
  gchar *launch = g_strdup_printf ("qtitofdepth calibration=\"%s\" %s", cal, props);
  GstHarness *h = gst_harness_new_parse (launch);

  g_free (launch);
  gst_harness_set_src_caps_str (h, TOF_DCS_CAPS);
  gst_harness_set_sink_caps_str (h,
      "video/x-raw, format=(string)GRAY16_LE, width=(int)240, height=(int)180, framerate=(fraction)30/1");
  return h;
}

GST_START_TEST (test_tof_references)
{
  gchar *cal = write_tmp ("[calibration]\nformat=2\n"
      "[far]\nR=4.0\nphi_ref=0\nc0=0\n[near]\nR=2.0\nphi_ref=0\nc0=0\n");
  GstHarness *h = props_harness (cal, "temporal-average=2");
  GstElement *el, *sink;
  GstPad *amp, *depth, *sinkpad;
  GstStructure *stats;
  RefCount cd = { 0 }, ca = { 0 };
  guint64 groups = 0, refs = 0, dropped = 0;

  /* averaging: equal amplitudes, so the mean of two phasors lies halfway */
  push_expect (h, timed (group_at (1, 0.5, 0.5), 0, FALSE), 500, 500, "near 0.5 m");
  push_expect (h, timed (group_at (1, 0.7, 0.7), 1, FALSE), 600, 600, "near 0.7 m, averaged with 0.5 m");
  push_expect (h, timed (group_at (0, 0.5, 2.5), 2, TRUE), 500, 2500, "reference, not averaged");
  push_expect (h, timed (group_at (1, 0.9, 0.9), 3, FALSE), 800, 800, "near 0.9 m, averaged with 0.7 m across it");
  push_expect (h, timed (group_at (0, 0.5, 2.5), 4, FALSE), 500, 2500, "unflagged far group, window restarted");
  push_expect (h, timed (group_at (1, 0.5, 0.5), 5, FALSE), 500, 500, "near after it, window restarted");
  gst_harness_teardown (h);

  /* output-references=false: the reference is used by unwrap but reaches no output, a GAP event instead */
  h = props_harness (cal, "unwrap=true output-references=false");
  el = gst_harness_find_element (h, "qtitofdepth");
  amp = gst_element_request_pad_simple (el, "amplitude");
  sink = gst_element_factory_make ("fakesink", NULL);
  g_object_set (sink, "sync", FALSE, NULL);
  sinkpad = gst_element_get_static_pad (sink, "sink");
  fail_if (gst_element_set_state (sink, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE);
  /* the harness keeps qtitofdepth in a bin of its own: link across the hierarchy */
  fail_unless_equals_int (gst_pad_link_full (amp, sinkpad, GST_PAD_LINK_CHECK_NOTHING), GST_PAD_LINK_OK);
  depth = gst_element_get_static_pad (el, "src");
  gst_pad_add_probe (depth, GST_PAD_PROBE_TYPE_BUFFER | GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM, ref_count_probe, &cd,
      NULL);
  gst_pad_add_probe (amp, GST_PAD_PROBE_TYPE_BUFFER | GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM, ref_count_probe, &ca, NULL);
  push_expect (h, timed (group_at (1, 0.5, 0.5), 0, FALSE), 500, 500, "near before the reference");
  fail_unless_equals_int (gst_harness_push (h, timed (group_at (0, 0.5, 2.5), 1, TRUE)), GST_FLOW_OK);
  fail_unless_equals_int (gst_harness_buffers_in_queue (h), 0);
  push_expect (h, timed (group_at (1, 0.5, 0.5), 2, FALSE), 500, 2500, "near after the left-out reference");
  fail_unless_equals_int (cd.buffers, 2);
  fail_unless_equals_int (cd.gaps, 1);
  fail_unless_equals_uint64 (cd.gap_ts, GST_SECOND / 30);
  fail_unless_equals_int (ca.buffers, 2);
  fail_unless_equals_int (ca.gaps, 1);
  fail_unless_equals_uint64 (ca.gap_ts, GST_SECOND / 30);
  g_object_get (el, "stats", &stats, NULL);
  fail_unless (gst_structure_get_uint64 (stats, "groups", &groups));
  fail_unless (gst_structure_get_uint64 (stats, "unwrap-references", &refs));
  fail_unless (gst_structure_get_uint64 (stats, "references-dropped", &dropped));
  fail_unless_equals_uint64 (groups, 3);
  fail_unless_equals_uint64 (refs, 1);
  fail_unless_equals_uint64 (dropped, 1);
  gst_structure_free (stats);
  gst_pad_unlink (amp, sinkpad);
  gst_element_set_state (sink, GST_STATE_NULL);
  gst_object_unref (sinkpad);
  gst_object_unref (sink);
  gst_object_unref (depth);
  gst_element_release_request_pad (el, amp);
  gst_object_unref (amp);
  gst_object_unref (el);
  gst_harness_teardown (h);

  /* default: the reference is output */
  h = props_harness (cal, "unwrap=true");
  push_expect (h, timed (group_at (1, 0.5, 0.5), 0, FALSE), 500, 500, "default: near");
  push_expect (h, timed (group_at (0, 0.5, 2.5), 1, TRUE), 500, 2500, "default: the reference is output");
  gst_harness_teardown (h);
  g_unlink (cal);
  g_free (cal);
}
GST_END_TEST;

/* as a group read by filesrc: no timestamps */
static GstBuffer *
untimed (GstBuffer * buf)
{
  GST_BUFFER_PTS (buf) = GST_BUFFER_DTS (buf) = GST_BUFFER_DURATION (buf) = GST_CLOCK_TIME_NONE;
  return buf;
}

/* push one group, check the timestamps of its depth output against group n of a 30/s stream */
static void
push_expect_ts (GstHarness * h, GstBuffer * in, guint64 n, const gchar * what)
{
  GstClockTime pts = gst_util_uint64_scale (n, GST_SECOND, 30);
  GstClockTime dur = gst_util_uint64_scale (n + 1, GST_SECOND, 30) - pts;
  GstBuffer *out;

  fail_unless_equals_int (gst_harness_push (h, in), GST_FLOW_OK);
  out = gst_harness_pull (h);
  fail_unless (out != NULL, "%s: no output", what);
  fail_unless (GST_BUFFER_PTS (out) == pts && GST_BUFFER_DURATION (out) == dur,
      "%s: pts %" GST_TIME_FORMAT " duration %" GST_TIME_FORMAT ", expected %" GST_TIME_FORMAT " %" GST_TIME_FORMAT,
      what, GST_TIME_ARGS (GST_BUFFER_PTS (out)), GST_TIME_ARGS (GST_BUFFER_DURATION (out)), GST_TIME_ARGS (pts),
      GST_TIME_ARGS (dur));
  gst_buffer_unref (out);
}

/* the next segment event the harness received: TIME, starting at start */
static void
expect_time_segment (GstHarness * h, GstClockTime start, const gchar * what)
{
  GstEvent *ev;

  while ((ev = gst_harness_try_pull_event (h))) {
    if (GST_EVENT_TYPE (ev) == GST_EVENT_SEGMENT) {
      const GstSegment *seg;

      gst_event_parse_segment (ev, &seg);
      fail_unless (seg->format == GST_FORMAT_TIME && seg->start == start && seg->time == start,
          "%s: %s segment start %" G_GUINT64_FORMAT ", expected TIME %" G_GUINT64_FORMAT, what,
          gst_format_get_name (seg->format), seg->start, start);
      gst_event_unref (ev);
      return;
    }
    gst_event_unref (ev);
  }
  fail ("%s: no segment", what);
}

/* A replay from a file runs in filesrc's BYTES segment with untimed groups: the outputs get a TIME segment from the
 * first group and timestamps from the group count, as aggregators such as qtivcomposer require */
GST_START_TEST (test_tof_bytes_segment)
{
  gchar *cal = write_tmp ("[calibration]\nformat=2\n"
      "[far]\nR=4.0\nphi_ref=0\nc0=0\n[near]\nR=2.0\nphi_ref=0\nc0=0\n");
  GstHarness *h = props_harness (cal, "");
  GstElement *el;
  GstPad *depth;
  GstSegment seg;
  RefCount cd = { 0 };

  gst_segment_init (&seg, GST_FORMAT_BYTES);
  fail_unless (gst_harness_push_event (h, gst_event_new_segment (&seg)));
  for (guint n = 0; n < 4; n++)
    push_expect_ts (h, untimed (group_at (1, 0.5, 0.5)), n, "BYTES from 0");
  expect_time_segment (h, 0, "BYTES from 0");
  /* a TIME segment again: the input timestamps are kept, as from qtitofsrc */
  gst_segment_init (&seg, GST_FORMAT_TIME);
  fail_unless (gst_harness_push_event (h, gst_event_new_segment (&seg)));
  push_expect_ts (h, timed (group_at (1, 0.5, 0.5), 7, FALSE), 7, "TIME after BYTES");
  expect_time_segment (h, 0, "TIME after BYTES");
  gst_harness_teardown (h);

  /* a BYTES seek to group 100 */
  h = props_harness (cal, "");
  gst_segment_init (&seg, GST_FORMAT_BYTES);
  seg.start = seg.time = seg.position = 100 * TOF_SIZE;
  fail_unless (gst_harness_push_event (h, gst_event_new_segment (&seg)));
  push_expect_ts (h, untimed (group_at (1, 0.5, 0.5)), 100, "BYTES from group 100");
  push_expect_ts (h, untimed (group_at (1, 0.5, 0.5)), 101, "BYTES from group 100");
  expect_time_segment (h, gst_util_uint64_scale (100, GST_SECOND, 30), "BYTES from group 100");
  gst_harness_teardown (h);

  /* a left-out reference: its GAP event at the group's time */
  h = props_harness (cal, "unwrap=true output-references=false");
  el = gst_harness_find_element (h, "qtitofdepth");
  depth = gst_element_get_static_pad (el, "src");
  gst_pad_add_probe (depth, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM, ref_count_probe, &cd, NULL);
  gst_segment_init (&seg, GST_FORMAT_BYTES);
  fail_unless (gst_harness_push_event (h, gst_event_new_segment (&seg)));
  push_expect_ts (h, untimed (group_at (1, 0.5, 0.5)), 0, "near before the reference");
  fail_unless_equals_int (gst_harness_push (h, untimed (timed (group_at (0, 0.5, 2.5), 0, TRUE))), GST_FLOW_OK);
  fail_unless_equals_int (gst_harness_buffers_in_queue (h), 0);
  push_expect_ts (h, untimed (group_at (1, 0.5, 0.5)), 2, "near after the left-out reference");
  fail_unless_equals_int (cd.gaps, 1);
  fail_unless_equals_uint64 (cd.gap_ts, gst_util_uint64_scale (1, GST_SECOND, 30));
  gst_object_unref (depth);
  gst_object_unref (el);
  gst_harness_teardown (h);
  g_unlink (cal);
  g_free (cal);
}
GST_END_TEST;

/* qtitofdepth in format, its sink proposing the video meta with 128-byte rows (as qtivcomposer does) or nothing */
static GstHarness *
align_harness (const gchar * cal, const gchar * format, gboolean ask)
{
  gchar *launch = g_strdup_printf ("qtitofdepth calibration=\"%s\"", cal);
  gchar *caps = g_strdup_printf ("video/x-raw, format=(string)%s, width=(int)240, height=(int)180, "
      "framerate=(fraction)30/1", format);
  GstHarness *h = gst_harness_new_parse (launch);

  if (ask) {
    GstStructure *params = gst_structure_new_empty ("video-meta");
    GstVideoAlignment align;

    gst_video_alignment_reset (&align);
    align.stride_align[0] = 127;
    gst_buffer_pool_config_set_video_alignment (params, &align);
    gst_harness_add_propose_allocation_meta (h, GST_VIDEO_META_API_TYPE, params);
    gst_structure_free (params);
  }
  gst_harness_set_src_caps_str (h, TOF_DCS_CAPS);
  gst_harness_set_sink_caps_str (h, caps);
  g_free (caps);
  g_free (launch);
  return h;
}

/* push one group at 0.5 m, check the row stride of the output's video meta; for GRAY16_LE also every pixel (500 mm) */
static void
expect_stride (GstHarness * h, gint stride, const gchar * what)
{
  GstBuffer *out;
  GstVideoMeta *meta;
  GstMapInfo map;

  fail_unless_equals_int (gst_harness_push (h, timed (group_at (1, 0.5, 0.5), 0, FALSE)), GST_FLOW_OK);
  out = gst_harness_pull (h);
  fail_unless (out != NULL, "%s: no output", what);
  meta = gst_buffer_get_video_meta (out);
  fail_unless (meta != NULL, "%s: no video meta", what);
  fail_unless (meta->stride[0] == stride, "%s: stride %d, expected %d", what, meta->stride[0], stride);
  gst_buffer_map (out, &map, GST_MAP_READ);
  fail_unless (map.size >= meta->offset[0] + (gsize) stride * TOF_H, "%s: %" G_GSIZE_FORMAT " bytes", what, map.size);
  if (meta->format == GST_VIDEO_FORMAT_GRAY16_LE)
    for (guint y = 0; y < TOF_H; y++)
      for (guint x = 0; x < TOF_W; x++) {
        gint got = GST_READ_UINT16_LE (map.data + meta->offset[0] + y * stride + 2 * x);

        fail_unless (ABS (got - 500) <= 3, "%s: pixel %u,%u %d mm, expected 500", what, x, y, got);
      }
  gst_buffer_unmap (out, &map);
  gst_buffer_unref (out);
}

/* GPU consumers such as qtivcomposer ask for a row alignment with the video meta; a tee passes it on without their
 * pool, and qtitofdepth's own pool follows it. Without such a request the rows keep the default layout */
GST_START_TEST (test_tof_alignment)
{
  gchar *cal = write_tmp ("[calibration]\nformat=2\n"
      "[far]\nR=4.0\nphi_ref=0\nc0=0\n[near]\nR=2.0\nphi_ref=0\nc0=0\n");
  static const struct
  {
    const gchar *format;
    gint plain, aligned;
  } cases[] = { {"GRAY8", 240, 256}, {"GRAY16_LE", 480, 512}, {"RGBA", 960, 1024} };

  for (guint i = 0; i < G_N_ELEMENTS (cases); i++) {
    gchar *what = g_strdup_printf ("%s, no request", cases[i].format);
    GstHarness *h = align_harness (cal, cases[i].format, FALSE);

    expect_stride (h, cases[i].plain, what);
    gst_harness_teardown (h);
    g_free (what);
    what = g_strdup_printf ("%s, 128-byte rows asked", cases[i].format);
    h = align_harness (cal, cases[i].format, TRUE);
    expect_stride (h, cases[i].aligned, what);
    gst_harness_teardown (h);
    g_free (what);
  }
  g_unlink (cal);
  g_free (cal);
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

  tcname = "calibration";
  tc = tcase_create (tcname);
  *tcnames = g_list_append (*tcnames, (gpointer) tcname);
  suite_add_tcase (s, tc);
  tcase_add_loop_test (tc, test_tof_calibration_default, start, end);

  tcname = "flying-pixels";
  tc = tcase_create (tcname);
  *tcnames = g_list_append (*tcnames, (gpointer) tcname);
  suite_add_tcase (s, tc);
  tcase_add_loop_test (tc, test_tof_flying_pixels, start, end);

  tcname = "depth-z";
  tc = tcase_create (tcname);
  *tcnames = g_list_append (*tcnames, (gpointer) tcname);
  suite_add_tcase (s, tc);
  tcase_add_loop_test (tc, test_tof_depth_z, start, end);

  tcname = "pointcloud";
  tc = tcase_create (tcname);
  *tcnames = g_list_append (*tcnames, (gpointer) tcname);
  suite_add_tcase (s, tc);
  tcase_add_loop_test (tc, test_tof_pointcloud, start, end);

  tcname = "wrap";
  tc = tcase_create (tcname);
  *tcnames = g_list_append (*tcnames, (gpointer) tcname);
  suite_add_tcase (s, tc);
  tcase_add_loop_test (tc, test_tof_wrap, start, end);

  tcname = "thread-hint";
  tc = tcase_create (tcname);
  *tcnames = g_list_append (*tcnames, (gpointer) tcname);
  suite_add_tcase (s, tc);
  tcase_add_loop_test (tc, test_tof_thread_hint, start, end);

  tcname = "unwrap";
  tc = tcase_create (tcname);
  *tcnames = g_list_append (*tcnames, (gpointer) tcname);
  suite_add_tcase (s, tc);
  tcase_add_loop_test (tc, test_tof_unwrap, start, end);

  tcname = "references";
  tc = tcase_create (tcname);
  *tcnames = g_list_append (*tcnames, (gpointer) tcname);
  suite_add_tcase (s, tc);
  tcase_add_loop_test (tc, test_tof_references, start, end);

  tcname = "bytes-segment";
  tc = tcase_create (tcname);
  *tcnames = g_list_append (*tcnames, (gpointer) tcname);
  suite_add_tcase (s, tc);
  tcase_add_loop_test (tc, test_tof_bytes_segment, start, end);

  tcname = "alignment";
  tc = tcase_create (tcname);
  *tcnames = g_list_append (*tcnames, (gpointer) tcname);
  suite_add_tcase (s, tc);
  tcase_add_loop_test (tc, test_tof_alignment, start, end);

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

  tcname = "far-reference";
  tc = tcase_create (tcname);
  *tcnames = g_list_append (*tcnames, (gpointer) tcname);
  suite_add_tcase (s, tc);
  tcase_set_timeout (tc, 40);
  tcase_add_loop_test (tc, test_tof_far_reference, start, end);

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
