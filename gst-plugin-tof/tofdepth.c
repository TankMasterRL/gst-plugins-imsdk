/*
 * Copyright (c) 2026, arducam-tof-ventuno project.
 * SPDX-License-Identifier: BSD-3-Clause-Clear
 *
 * qtitofdepth: DCS16 groups (qtitofsrc) -> calibrated depth, amplitude and confidence (DEC-0020,
 * wiki/ToF-Elements.md).
 *
 * Per pixel I = F0 - F2, Q = F1 - F3, A = |I + jQ| / 2, phi = atan2 (Q, I) (no offsets, DEC-0018).
 * The calibration (GLib key file written by tools/tof-sweep.py) gives per range mode R, phi_ref and
 * y(d) = c0 [+ c1 d] + sum_k a_k cos (k theta) + b_k sin (k theta), theta = 2 pi d / R; the radial
 * distance solves d = R / 2pi * (u + y (d)), u = wrap (phi_ref - phi) in [0, 2pi). y depends on d only,
 * so the equation is solved once per mode at load time (fixed-point iteration, as tof-sweep.py) into a
 * table over u; a pixel costs one atan2 and one interpolated lookup.
 *
 * Pads: "src" (always) depth, GRAY16_LE millimetres or a colour-mapped view; "amplitude" and
 * "confidence" (request). Outputs not linked are not computed. Invalid pixels (amplitude below
 * min-amplitude, or a DCS sample at the 11-bit rails) are 0. Output buffers come from the downstream
 * image pool or from a QIM image pool with the QTI DMA-buf allocator (videotransform.c pattern).
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "tofdepth.h"
#include "tofdcs.h"

#include <math.h>
#include <string.h>

#include <gst/base/gstflowcombiner.h>
#include <gst/video/video.h>
#include <gst/allocators/gstqtiallocator.h>
#include <gst/video/gstimagepool.h>

GST_DEBUG_CATEGORY_STATIC (gst_tof_depth_debug);
#define GST_CAT_DEFAULT gst_tof_depth_debug

#define N_PIX            (GST_TOF_WIDTH * GST_TOF_HEIGHT)
#define LUT_SIZE         4096
#define MAX_HARMONICS    4
#define MAX_AVERAGE      16
#define TWO_PI           (2.0 * G_PI)

#define DEFAULT_MIN_AMPLITUDE    3.0
#define DEFAULT_MIN_DISTANCE     0.0
#define DEFAULT_MAX_DISTANCE     0.0
#define DEFAULT_AMPLITUDE_MAX    255.0
#define DEFAULT_CONFIDENCE_SCALE 64.0
#define DEFAULT_TEMPORAL_AVERAGE 1

#define DEPTH_FORMATS "{ GRAY16_LE, RGBA, BGRA, RGB, BGR, GRAY8 }"
#define AMPLITUDE_FORMATS "{ GRAY16_LE, GRAY8 }"
#define CONFIDENCE_FORMATS "{ GRAY8, GRAY16_LE }"
#define VIDEO_CAPS(formats) \
  "video/x-raw, format=(string)" formats ", width=(int)240, height=(int)180, " \
  "framerate=(fraction)[ 0/1, 1000/1 ]"

typedef enum
{
  GST_TOF_COLORMAP_TURBO,
  GST_TOF_COLORMAP_GRAY,
} GstTofColormap;

typedef enum
{
  GST_TOF_DEPTH_RADIAL,
} GstTofDepthType;

enum
{
  OUT_DEPTH,
  OUT_AMPLITUDE,
  OUT_CONFIDENCE,
  N_OUT
};

enum
{
  PROP_0,
  PROP_CALIBRATION,
  PROP_MIN_AMPLITUDE,
  PROP_MIN_DISTANCE,
  PROP_MAX_DISTANCE,
  PROP_COLORMAP,
  PROP_AMPLITUDE_MAX,
  PROP_CONFIDENCE_SCALE,
  PROP_TEMPORAL_AVERAGE,
  PROP_DEPTH_TYPE,
  PROP_STATS,
};

typedef struct
{
  gdouble r;
  gdouble phi_ref;
  gdouble c0, c1;
  gdouble a[MAX_HARMONICS], b[MAX_HARMONICS];   /* terms of 4 (h + 1) theta */
  gfloat lut[LUT_SIZE + 1];                     /* unwrapped distance (m) over u */
  gboolean calibrated;
} TofModel;

typedef struct
{
  GstPad *pad;
  GstVideoInfo info;
  gboolean need_caps;
  gboolean need_stream_start;
  gboolean need_segment;
  GstBufferPool *pool;
} TofOut;

struct _GstTofDepth
{
  GstElement parent;

  GstPad *sinkpad;
  TofOut outs[N_OUT];
  GstFlowCombiner *combiner;

  /* properties (object lock) */
  gchar *calibration;
  gdouble min_amplitude;
  gdouble min_distance;
  gdouble max_distance;
  GstTofColormap colormap;
  gdouble amplitude_max;
  gdouble confidence_scale;
  guint temporal_average;

  /* stream state (streaming thread) */
  TofModel models[2];
  gboolean models_loaded;
  GstEvent *stream_start;
  GstSegment segment;
  gint fps_n, fps_d;
  guint8 turbo[256][3];
  gfloat *dist;                 /* metres, 0 = invalid */
  gfloat *amp;
  gfloat *sum_i, *sum_q, *hist;
  guint hist_count, hist_pos, hist_n;
  gint hist_mode;

  /* stats (object lock) */
  guint64 n_groups;
  guint64 proc_ns;
};

#define gst_tof_depth_parent_class parent_class
G_DEFINE_TYPE (GstTofDepth, gst_tof_depth, GST_TYPE_ELEMENT);

static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS (GST_TOF_DCS_CAPS));
static GstStaticPadTemplate depth_template = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS (VIDEO_CAPS (DEPTH_FORMATS)));
static GstStaticPadTemplate amplitude_template = GST_STATIC_PAD_TEMPLATE ("amplitude",
    GST_PAD_SRC, GST_PAD_REQUEST, GST_STATIC_CAPS (VIDEO_CAPS (AMPLITUDE_FORMATS)));
static GstStaticPadTemplate confidence_template = GST_STATIC_PAD_TEMPLATE ("confidence",
    GST_PAD_SRC, GST_PAD_REQUEST, GST_STATIC_CAPS (VIDEO_CAPS (CONFIDENCE_FORMATS)));

static GType
gst_tof_colormap_get_type (void)
{
  static gsize type = 0;
  static const GEnumValue values[] = {
    { GST_TOF_COLORMAP_TURBO, "Turbo colour map, near = red", "turbo" },
    { GST_TOF_COLORMAP_GRAY, "Grey, near = bright", "gray" },
    { 0, NULL, NULL },
  };

  if (g_once_init_enter (&type)) {
    GType t = g_enum_register_static ("GstTofColormap", values);
    g_once_init_leave (&type, t);
  }
  return type;
}

static GType
gst_tof_depth_type_get_type (void)
{
  static gsize type = 0;
  static const GEnumValue values[] = {
    { GST_TOF_DEPTH_RADIAL, "Radial distance along each pixel's ray (z needs the lens model, T6.1)",
        "radial" },
    { 0, NULL, NULL },
  };

  if (g_once_init_enter (&type)) {
    GType t = g_enum_register_static ("GstTofDepthType", values);
    g_once_init_leave (&type, t);
  }
  return type;
}

/* ---------------------------------------------------------------- model */

static gdouble
model_y (const TofModel * m, gdouble d)
{
  gdouble th = TWO_PI * d / m->r, y = m->c0 + m->c1 * d;

  for (guint h = 0; h < MAX_HARMONICS; h++)
    if (m->a[h] != 0.0 || m->b[h] != 0.0)
      y += m->a[h] * cos (4.0 * (h + 1) * th) + m->b[h] * sin (4.0 * (h + 1) * th);
  return y;
}

static void
model_build_lut (TofModel * m)
{
  for (guint k = 0; k <= LUT_SIZE; k++) {
    gdouble u = TWO_PI * k / LUT_SIZE;
    gdouble d = m->r * (u + m->c0) / TWO_PI;

    for (guint it = 0; it < 30; it++)
      d = m->r * (u + model_y (m, d)) / TWO_PI;
    m->lut[k] = (gfloat) d;
  }
}

static void
model_init_nominal (TofModel * m, gdouble r)
{
  memset (m, 0, sizeof (*m));
  m->r = r;
  model_build_lut (m);
}

static gboolean
model_load (GstTofDepth * self, GKeyFile * kf, const gchar * group, TofModel * m)
{
  GError *err = NULL;
  gchar key[8];

  memset (m, 0, sizeof (*m));
  m->r = g_key_file_get_double (kf, group, "R", &err);
  if (!err)
    m->phi_ref = g_key_file_get_double (kf, group, "phi_ref", &err);
  if (!err)
    m->c0 = g_key_file_get_double (kf, group, "c0", &err);
  if (err) {
    GST_ERROR_OBJECT (self, "calibration [%s]: %s", group, err->message);
    g_error_free (err);
    return FALSE;
  }
  if (g_key_file_has_key (kf, group, "c1", NULL))
    m->c1 = g_key_file_get_double (kf, group, "c1", NULL);
  for (guint h = 0; h < MAX_HARMONICS; h++) {
    g_snprintf (key, sizeof (key), "a%u", 4 * (h + 1));
    if (g_key_file_has_key (kf, group, key, NULL))
      m->a[h] = g_key_file_get_double (kf, group, key, NULL);
    g_snprintf (key, sizeof (key), "b%u", 4 * (h + 1));
    if (g_key_file_has_key (kf, group, key, NULL))
      m->b[h] = g_key_file_get_double (kf, group, key, NULL);
  }
  if (g_key_file_has_key (kf, group, "oI", NULL) || g_key_file_has_key (kf, group, "oQ", NULL))
    GST_WARNING_OBJECT (self, "calibration [%s]: I/Q offset terms are not supported, ignored", group);
  if (m->r <= 0.0) {
    GST_ERROR_OBJECT (self, "calibration [%s]: R must be > 0", group);
    return FALSE;
  }
  m->calibrated = TRUE;
  model_build_lut (m);
  GST_INFO_OBJECT (self, "[%s] R %.1f m, phi_ref %+.4f, c0 %+.4f, a4 %+.4f, b4 %+.4f", group,
      m->r, m->phi_ref, m->c0, m->a[0], m->b[0]);
  return TRUE;
}

static gboolean
load_models (GstTofDepth * self)
{
  GKeyFile *kf;
  GError *err = NULL;
  gchar *path;
  gboolean ok;

  GST_OBJECT_LOCK (self);
  path = g_strdup (self->calibration);
  GST_OBJECT_UNLOCK (self);

  model_init_nominal (&self->models[GST_TOF_RANGE_FAR], 4.0);
  model_init_nominal (&self->models[GST_TOF_RANGE_NEAR], 2.0);
  self->models_loaded = TRUE;
  if (!path) {
    GST_ELEMENT_WARNING (self, RESOURCE, SETTINGS, ("No calibration: depth is not metric."),
        ("Set calibration= to a key file written by tools/tof-sweep.py (CAL_OUT=*.conf)."));
    return TRUE;
  }
  kf = g_key_file_new ();
  ok = g_key_file_load_from_file (kf, path, G_KEY_FILE_NONE, &err) &&
      model_load (self, kf, "far", &self->models[GST_TOF_RANGE_FAR]) &&
      model_load (self, kf, "near", &self->models[GST_TOF_RANGE_NEAR]);
  if (err) {
    GST_ELEMENT_ERROR (self, RESOURCE, OPEN_READ, ("Cannot read calibration %s.", path),
        ("%s", err->message));
    g_error_free (err);
  } else if (!ok) {
    GST_ELEMENT_ERROR (self, RESOURCE, SETTINGS, ("Invalid calibration %s.", path), (NULL));
  }
  g_key_file_free (kf);
  g_free (path);
  return ok;
}

/* Turbo colour map, polynomial approximation (A. Mikhailov, Google, 2019, Apache-2.0) */
static void
build_turbo (guint8 lut[256][3])
{
  for (guint i = 0; i < 256; i++) {
    gdouble x = i / 255.0, c[3];

    c[0] = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 +
                x * (-152.94239396 + x * 59.28637943))));
    c[1] = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 +
                x * (4.27729857 + x * 2.82956604))));
    c[2] = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 +
                x * (-89.90310912 + x * 27.34824973))));
    for (guint k = 0; k < 3; k++)
      lut[i][k] = (guint8) CLAMP (lround (c[k] * 255.0), 0, 255);
  }
}

/* ---------------------------------------------------------------- computation */

static void
compute (GstTofDepth * self, const guint8 * data, gboolean want_depth)
{
  const gint16 *f0 = (const gint16 *) (data + GST_TOF_HEADER_SIZE);
  const gint16 *f1 = f0 + N_PIX, *f2 = f1 + N_PIX, *f3 = f2 + N_PIX;
  guint mode = data[GST_TOF_OFF_MODE] ? GST_TOF_RANGE_NEAR : GST_TOF_RANGE_FAR;
  const TofModel *m = &self->models[mode];
  gfloat min_amp, ref = (gfloat) m->phi_ref, r = (gfloat) m->r;
  guint navg;
  gfloat *ci = NULL, *cq = NULL;

  GST_OBJECT_LOCK (self);
  min_amp = (gfloat) self->min_amplitude;
  navg = self->temporal_average;
  GST_OBJECT_UNLOCK (self);

  if (navg > 1) {
    /* sliding window over navg groups of the same mode */
    if (!self->hist || self->hist_n != navg || self->hist_mode != (gint) mode) {
      g_free (self->hist);
      self->hist = g_new0 (gfloat, (gsize) navg * 2 * N_PIX);
      memset (self->sum_i, 0, N_PIX * sizeof (gfloat));
      memset (self->sum_q, 0, N_PIX * sizeof (gfloat));
      self->hist_n = navg;
      self->hist_count = self->hist_pos = 0;
      self->hist_mode = mode;
    }
    ci = self->hist + (gsize) self->hist_pos * 2 * N_PIX;
    cq = ci + N_PIX;
    for (guint p = 0; p < N_PIX; p++) {
      gfloat i = (gfloat) (GINT16_FROM_LE (f0[p]) - GINT16_FROM_LE (f2[p]));
      gfloat q = (gfloat) (GINT16_FROM_LE (f1[p]) - GINT16_FROM_LE (f3[p]));

      if (self->hist_count == navg) {
        self->sum_i[p] -= ci[p];
        self->sum_q[p] -= cq[p];
      }
      ci[p] = i;
      cq[p] = q;
      self->sum_i[p] += i;
      self->sum_q[p] += q;
    }
    if (self->hist_count < navg)
      self->hist_count++;
    self->hist_pos = (self->hist_pos + 1) % navg;
  }

  for (guint p = 0; p < N_PIX; p++) {
    gint s0 = GINT16_FROM_LE (f0[p]), s1 = GINT16_FROM_LE (f1[p]);
    gint s2 = GINT16_FROM_LE (f2[p]), s3 = GINT16_FROM_LE (f3[p]);
    gfloat i, q, a;
    gboolean rail = s0 >= 1023 || s0 <= -1024 || s1 >= 1023 || s1 <= -1024 ||
        s2 >= 1023 || s2 <= -1024 || s3 >= 1023 || s3 <= -1024;

    if (navg > 1) {
      i = self->sum_i[p] / self->hist_count;
      q = self->sum_q[p] / self->hist_count;
    } else {
      i = (gfloat) (s0 - s2);
      q = (gfloat) (s1 - s3);
    }
    a = sqrtf (i * i + q * q) * 0.5f;
    self->amp[p] = a;
    if (rail || a < min_amp) {
      self->dist[p] = 0.0f;
      continue;
    }
    if (want_depth) {
      gfloat u = ref - atan2f (q, i), x, d;
      guint k;

      u -= (gfloat) TWO_PI * floorf (u / (gfloat) TWO_PI);
      x = u * (LUT_SIZE / (gfloat) TWO_PI);
      k = MIN ((guint) x, LUT_SIZE - 1);
      d = m->lut[k] + (x - k) * (m->lut[k + 1] - m->lut[k]);
      d -= r * floorf (d / r);
      self->dist[p] = d > 0.0f ? d : 1e-4f;
    } else {
      self->dist[p] = 1.0f;     /* valid marker only */
    }
  }
}

static void
fill_depth (GstTofDepth * self, GstVideoFrame * vf, guint mode)
{
  GstVideoFormat fmt = GST_VIDEO_FRAME_FORMAT (vf);
  guint8 *base = GST_VIDEO_FRAME_PLANE_DATA (vf, 0);
  gint stride = GST_VIDEO_FRAME_PLANE_STRIDE (vf, 0);
  gfloat lo, hi, span;
  GstTofColormap cmap;

  GST_OBJECT_LOCK (self);
  lo = (gfloat) self->min_distance;
  hi = self->max_distance > 0.0 ? (gfloat) self->max_distance : (gfloat) self->models[mode].r;
  cmap = self->colormap;
  GST_OBJECT_UNLOCK (self);
  span = hi > lo ? hi - lo : 1.0f;

  for (guint y = 0; y < GST_TOF_HEIGHT; y++) {
    guint8 *row = base + y * stride;
    const gfloat *d = self->dist + y * GST_TOF_WIDTH;

    for (guint x = 0; x < GST_TOF_WIDTH; x++) {
      if (fmt == GST_VIDEO_FORMAT_GRAY16_LE) {
        guint v = d[x] > 0.0f ? (guint) CLAMP (lrintf (d[x] * 1000.0f), 1, 65535) : 0;

        GST_WRITE_UINT16_LE (row + 2 * x, v);
        continue;
      }
      /* near = bright / red */
      gint idx = d[x] > 0.0f ? (gint) CLAMP (lrintf (255.0f * (1.0f - (d[x] - lo) / span)), 0, 255) : -1;

      switch (fmt) {
        case GST_VIDEO_FORMAT_GRAY8:
          row[x] = idx < 0 ? 0 : (guint8) idx;
          break;
        case GST_VIDEO_FORMAT_RGBA:
        case GST_VIDEO_FORMAT_BGRA:
        case GST_VIDEO_FORMAT_RGB:
        case GST_VIDEO_FORMAT_BGR:{
          guint bpp = (fmt == GST_VIDEO_FORMAT_RGBA || fmt == GST_VIDEO_FORMAT_BGRA) ? 4 : 3;
          gboolean bgr = fmt == GST_VIDEO_FORMAT_BGRA || fmt == GST_VIDEO_FORMAT_BGR;
          guint8 c[3] = { 0, 0, 0 }, *px = row + bpp * x;

          if (idx >= 0) {
            if (cmap == GST_TOF_COLORMAP_GRAY)
              c[0] = c[1] = c[2] = (guint8) idx;
            else
              memcpy (c, self->turbo[idx], 3);
          }
          px[0] = bgr ? c[2] : c[0];
          px[1] = c[1];
          px[2] = bgr ? c[0] : c[2];
          if (bpp == 4)
            px[3] = 255;
          break;
        }
        default:
          break;
      }
    }
  }
}

static void
fill_scalar (GstTofDepth * self, GstVideoFrame * vf, gboolean confidence)
{
  GstVideoFormat fmt = GST_VIDEO_FRAME_FORMAT (vf);
  guint8 *base = GST_VIDEO_FRAME_PLANE_DATA (vf, 0);
  gint stride = GST_VIDEO_FRAME_PLANE_STRIDE (vf, 0);
  gfloat amax, cscale;

  GST_OBJECT_LOCK (self);
  amax = (gfloat) self->amplitude_max;
  cscale = (gfloat) self->confidence_scale;
  GST_OBJECT_UNLOCK (self);

  for (guint y = 0; y < GST_TOF_HEIGHT; y++) {
    guint8 *row = base + y * stride;
    const gfloat *a = self->amp + y * GST_TOF_WIDTH, *d = self->dist + y * GST_TOF_WIDTH;

    for (guint x = 0; x < GST_TOF_WIDTH; x++) {
      if (confidence) {
        gfloat c = d[x] > 0.0f ? MIN (1.0f, a[x] / cscale) : 0.0f;

        if (fmt == GST_VIDEO_FORMAT_GRAY8)
          row[x] = d[x] > 0.0f ? (guint8) CLAMP (lrintf (255.0f * c), 1, 255) : 0;
        else
          GST_WRITE_UINT16_LE (row + 2 * x,
              d[x] > 0.0f ? (guint16) CLAMP (lrintf (65535.0f * c), 1, 65535) : 0);
      } else {
        if (fmt == GST_VIDEO_FORMAT_GRAY8)
          row[x] = (guint8) CLAMP (lrintf (255.0f * a[x] / amax), 0, 255);
        else
          GST_WRITE_UINT16_LE (row + 2 * x, (guint16) CLAMP (lrintf (a[x]), 0, 65535));
      }
    }
  }
}

/* ---------------------------------------------------------------- negotiation and pools */

static GstBufferPool *
create_pool (GstTofDepth * self, GstCaps * caps, GstVideoInfo * info)
{
  GstBufferPool *pool = gst_image_buffer_pool_new ();
  GstAllocator *allocator = gst_qti_allocator_new (GST_FD_MEMORY_FLAG_KEEP_MAPPED);
  GstStructure *config;

  if (!pool || !allocator) {
    GST_WARNING_OBJECT (self, "no QIM image pool / QTI allocator, using a video pool");
    gst_clear_object (&pool);
    gst_clear_object (&allocator);
    pool = gst_video_buffer_pool_new ();
    config = gst_buffer_pool_get_config (pool);
  } else {
    config = gst_buffer_pool_get_config (pool);
    gst_buffer_pool_config_set_allocator (config, allocator, NULL);
    gst_object_unref (allocator);
    gst_buffer_pool_config_add_option (config, GST_IMAGE_BUFFER_POOL_OPTION_KEEP_MAPPED);
  }
  gst_buffer_pool_config_add_option (config, GST_BUFFER_POOL_OPTION_VIDEO_META);
  gst_buffer_pool_config_set_params (config, caps, info->size, 2, 0);
  if (!gst_buffer_pool_set_config (pool, config)) {
    GST_ERROR_OBJECT (self, "pool configuration rejected");
    gst_clear_object (&pool);
  }
  return pool;
}

static gboolean
push_sticky (GstTofDepth * self, TofOut * out)
{
  if (out->need_stream_start && self->stream_start) {
    const gchar *upstream_id = NULL;
    gchar *id;
    GstEvent *ev;
    guint group;

    gst_event_parse_stream_start (self->stream_start, &upstream_id);
    id = g_strdup_printf ("%s/%s", GST_STR_NULL (upstream_id), GST_PAD_NAME (out->pad));
    ev = gst_event_new_stream_start (id);
    if (gst_event_parse_group_id (self->stream_start, &group))
      gst_event_set_group_id (ev, group);
    g_free (id);
    gst_pad_push_event (out->pad, ev);
    out->need_stream_start = FALSE;
  }
  if (out->need_caps) {
    GstCaps *tmpl = gst_pad_get_pad_template_caps (out->pad), *peer, *caps;
    GstQuery *query;

    tmpl = gst_caps_make_writable (tmpl);
    gst_caps_set_simple (tmpl, "framerate", GST_TYPE_FRACTION, self->fps_n, self->fps_d, NULL);
    peer = gst_pad_peer_query_caps (out->pad, tmpl);
    gst_caps_unref (tmpl);
    if (gst_caps_is_empty (peer)) {
      gst_caps_unref (peer);
      GST_ELEMENT_ERROR (self, CORE, NEGOTIATION, ("No common format on %s.",
              GST_PAD_NAME (out->pad)), (NULL));
      return FALSE;
    }
    caps = gst_caps_fixate (peer);
    if (!gst_video_info_from_caps (&out->info, caps) || !gst_pad_push_event (out->pad,
            gst_event_new_caps (caps))) {
      gst_caps_unref (caps);
      GST_ELEMENT_ERROR (self, CORE, NEGOTIATION, ("Caps not accepted on %s.",
              GST_PAD_NAME (out->pad)), (NULL));
      return FALSE;
    }
    GST_INFO_OBJECT (out->pad, "caps %" GST_PTR_FORMAT, caps);

    if (out->pool) {
      gst_buffer_pool_set_active (out->pool, FALSE);
      gst_clear_object (&out->pool);
    }
    query = gst_query_new_allocation (caps, TRUE);
    if (gst_pad_peer_query (out->pad, query) && gst_query_get_n_allocation_pools (query) > 0) {
      GstBufferPool *pool = NULL;
      guint size, min, max;

      gst_query_parse_nth_allocation_pool (query, 0, &pool, &size, &min, &max);
      if (pool && GST_IS_IMAGE_BUFFER_POOL (pool)) {
        GstStructure *config = gst_buffer_pool_get_config (pool);

        gst_buffer_pool_config_set_params (config, caps, MAX (size, out->info.size), MAX (min, 2), max);
        gst_buffer_pool_config_add_option (config, GST_BUFFER_POOL_OPTION_VIDEO_META);
        if (gst_buffer_pool_set_config (pool, config))
          out->pool = pool;
        else
          gst_object_unref (pool);
      } else if (pool) {
        gst_object_unref (pool);
      }
    }
    gst_query_unref (query);
    if (!out->pool)
      out->pool = create_pool (self, caps, &out->info);
    gst_caps_unref (caps);
    if (!out->pool || !gst_buffer_pool_set_active (out->pool, TRUE)) {
      GST_ELEMENT_ERROR (self, RESOURCE, NO_SPACE_LEFT, ("Cannot activate the buffer pool of %s.",
              GST_PAD_NAME (out->pad)), (NULL));
      return FALSE;
    }
    out->need_caps = FALSE;
  }
  if (out->need_segment) {
    gst_pad_push_event (out->pad, gst_event_new_segment (&self->segment));
    out->need_segment = FALSE;
  }
  return TRUE;
}

static GstFlowReturn
push_output (GstTofDepth * self, TofOut * out, guint idx, guint mode, GstBuffer * inbuf)
{
  GstBuffer *buf = NULL;
  GstVideoFrame vf;
  GstFlowReturn ret;

  if (!push_sticky (self, out))
    return GST_FLOW_NOT_NEGOTIATED;
  ret = gst_buffer_pool_acquire_buffer (out->pool, &buf, NULL);
  if (ret != GST_FLOW_OK)
    return ret;
  if (!gst_video_frame_map (&vf, &out->info, buf, GST_MAP_WRITE)) {
    gst_buffer_unref (buf);
    GST_ELEMENT_ERROR (self, RESOURCE, WRITE, ("Cannot map an output buffer."), (NULL));
    return GST_FLOW_ERROR;
  }
  if (idx == OUT_DEPTH)
    fill_depth (self, &vf, mode);
  else
    fill_scalar (self, &vf, idx == OUT_CONFIDENCE);
  gst_video_frame_unmap (&vf);
  gst_buffer_copy_into (buf, inbuf, GST_BUFFER_COPY_TIMESTAMPS | GST_BUFFER_COPY_FLAGS, 0, -1);
  return gst_pad_push (out->pad, buf);
}

/* ---------------------------------------------------------------- pads */

static GstFlowReturn
gst_tof_depth_chain (GstPad * pad, GstObject * parent, GstBuffer * inbuf)
{
  GstTofDepth *self = GST_TOF_DEPTH (parent);
  GstFlowReturn ret = GST_FLOW_OK;
  GstMapInfo map;
  gboolean active[N_OUT];
  guint mode;
  guint64 t0 = g_get_monotonic_time ();

  if (!self->models_loaded && !load_models (self)) {
    gst_buffer_unref (inbuf);
    return GST_FLOW_ERROR;
  }
  if (!gst_buffer_map (inbuf, &map, GST_MAP_READ))
    goto map_failed;
  if (map.size < GST_TOF_BUFFER_SIZE || GST_READ_UINT32_LE (map.data) != GST_TOF_MAGIC) {
    gst_buffer_unmap (inbuf, &map);
    GST_ELEMENT_ERROR (self, STREAM, FORMAT, ("Not a DCS16 group (%" G_GSIZE_FORMAT " bytes).",
            map.size), (NULL));
    gst_buffer_unref (inbuf);
    return GST_FLOW_ERROR;
  }
  mode = map.data[GST_TOF_OFF_MODE] ? GST_TOF_RANGE_NEAR : GST_TOF_RANGE_FAR;

  GST_OBJECT_LOCK (self);
  for (guint k = 0; k < N_OUT; k++)
    active[k] = self->outs[k].pad && gst_pad_is_linked (self->outs[k].pad);
  GST_OBJECT_UNLOCK (self);

  compute (self, map.data, active[OUT_DEPTH] || active[OUT_CONFIDENCE]);
  gst_buffer_unmap (inbuf, &map);

  for (guint k = 0; k < N_OUT; k++) {
    GstFlowReturn r;

    if (!active[k])
      continue;
    r = push_output (self, &self->outs[k], k, mode, inbuf);
    ret = gst_flow_combiner_update_pad_flow (self->combiner, self->outs[k].pad, r);
    if (ret != GST_FLOW_OK && ret != GST_FLOW_NOT_LINKED)
      break;
  }
  if (!active[OUT_DEPTH] && !active[OUT_AMPLITUDE] && !active[OUT_CONFIDENCE])
    ret = GST_FLOW_NOT_LINKED;

  GST_OBJECT_LOCK (self);
  self->n_groups++;
  self->proc_ns += (g_get_monotonic_time () - t0) * 1000;
  GST_OBJECT_UNLOCK (self);
  gst_buffer_unref (inbuf);
  return ret;

map_failed:
  gst_buffer_unref (inbuf);
  GST_ELEMENT_ERROR (self, RESOURCE, READ, ("Cannot map the input buffer."), (NULL));
  return GST_FLOW_ERROR;
}

static void
forward_event (GstTofDepth * self, GstEvent * event)
{
  for (guint k = 0; k < N_OUT; k++) {
    TofOut *out = &self->outs[k];

    if (!out->pad)
      continue;
    if (GST_EVENT_TYPE (event) == GST_EVENT_EOS) {
      if (gst_pad_is_linked (out->pad) && !push_sticky (self, out))
        continue;
    } else if (GST_EVENT_IS_STICKY (event) &&
        (out->need_stream_start || out->need_caps || out->need_segment)) {
      /* a sticky event (tags, ...) before this output is set up would be out of order */
      GST_DEBUG_OBJECT (out->pad, "dropping %" GST_PTR_FORMAT " before set-up", event);
      continue;
    }
    gst_pad_push_event (out->pad, gst_event_ref (event));
  }
  gst_event_unref (event);
}

static gboolean
gst_tof_depth_sink_event (GstPad * pad, GstObject * parent, GstEvent * event)
{
  GstTofDepth *self = GST_TOF_DEPTH (parent);

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_STREAM_START:
      gst_event_replace (&self->stream_start, event);
      for (guint k = 0; k < N_OUT; k++)
        self->outs[k].need_stream_start = TRUE;
      gst_event_unref (event);
      return TRUE;
    case GST_EVENT_CAPS:{
      GstCaps *caps;
      GstStructure *s;

      gst_event_parse_caps (event, &caps);
      s = gst_caps_get_structure (caps, 0);
      if (!gst_structure_get_fraction (s, "framerate", &self->fps_n, &self->fps_d)) {
        self->fps_n = 30;
        self->fps_d = 1;
      }
      for (guint k = 0; k < N_OUT; k++)
        self->outs[k].need_caps = TRUE;
      gst_event_unref (event);
      return TRUE;
    }
    case GST_EVENT_SEGMENT:
      gst_event_copy_segment (event, &self->segment);
      for (guint k = 0; k < N_OUT; k++)
        self->outs[k].need_segment = TRUE;
      gst_event_unref (event);
      return TRUE;
    case GST_EVENT_FLUSH_STOP:
      gst_flow_combiner_reset (self->combiner);
      self->hist_count = self->hist_pos = 0;
      if (self->sum_i) {
        memset (self->sum_i, 0, N_PIX * sizeof (gfloat));
        memset (self->sum_q, 0, N_PIX * sizeof (gfloat));
      }
      forward_event (self, event);
      return TRUE;
    default:
      forward_event (self, event);
      return TRUE;
  }
}

static gboolean
gst_tof_depth_src_query (GstPad * pad, GstObject * parent, GstQuery * query)
{
  if (GST_QUERY_TYPE (query) == GST_QUERY_CAPS) {
    GstCaps *filter, *caps = gst_pad_get_pad_template_caps (pad);

    gst_query_parse_caps (query, &filter);
    if (filter) {
      GstCaps *tmp = gst_caps_intersect_full (filter, caps, GST_CAPS_INTERSECT_FIRST);

      gst_caps_unref (caps);
      caps = tmp;
    }
    gst_query_set_caps_result (query, caps);
    gst_caps_unref (caps);
    return TRUE;
  }
  return gst_pad_query_default (pad, parent, query);
}

static gboolean
gst_tof_depth_src_event (GstPad * pad, GstObject * parent, GstEvent * event)
{
  if (GST_EVENT_TYPE (event) == GST_EVENT_RECONFIGURE) {
    GstTofDepth *self = GST_TOF_DEPTH (parent);

    for (guint k = 0; k < N_OUT; k++)
      if (self->outs[k].pad == pad)
        self->outs[k].need_caps = TRUE;
    gst_event_unref (event);
    return TRUE;
  }
  return gst_pad_event_default (pad, parent, event);
}

static void
setup_src_pad (GstTofDepth * self, TofOut * out, GstPad * pad)
{
  out->pad = pad;
  out->need_caps = out->need_stream_start = out->need_segment = TRUE;
  gst_pad_set_query_function (pad, GST_DEBUG_FUNCPTR (gst_tof_depth_src_query));
  gst_pad_set_event_function (pad, GST_DEBUG_FUNCPTR (gst_tof_depth_src_event));
}

static GstPad *
gst_tof_depth_request_new_pad (GstElement * element, GstPadTemplate * templ,
    const gchar * name, const GstCaps * caps)
{
  GstTofDepth *self = GST_TOF_DEPTH (element);
  const gchar *tname = GST_PAD_TEMPLATE_NAME_TEMPLATE (templ);
  guint idx = g_strcmp0 (tname, "amplitude") == 0 ? OUT_AMPLITUDE : OUT_CONFIDENCE;
  GstPad *pad;

  GST_OBJECT_LOCK (self);
  if (self->outs[idx].pad) {
    GST_OBJECT_UNLOCK (self);
    GST_ERROR_OBJECT (self, "pad %s already exists", tname);
    return NULL;
  }
  GST_OBJECT_UNLOCK (self);
  pad = gst_pad_new_from_template (templ, tname);
  GST_OBJECT_LOCK (self);
  setup_src_pad (self, &self->outs[idx], pad);
  GST_OBJECT_UNLOCK (self);
  gst_pad_set_active (pad, TRUE);
  gst_element_add_pad (element, pad);
  gst_flow_combiner_add_pad (self->combiner, pad);
  return pad;
}

static void
gst_tof_depth_release_pad (GstElement * element, GstPad * pad)
{
  GstTofDepth *self = GST_TOF_DEPTH (element);

  GST_OBJECT_LOCK (self);
  for (guint k = OUT_AMPLITUDE; k < N_OUT; k++) {
    if (self->outs[k].pad == pad) {
      self->outs[k].pad = NULL;
      if (self->outs[k].pool) {
        gst_buffer_pool_set_active (self->outs[k].pool, FALSE);
        gst_clear_object (&self->outs[k].pool);
      }
    }
  }
  GST_OBJECT_UNLOCK (self);
  gst_flow_combiner_remove_pad (self->combiner, pad);
  gst_element_remove_pad (element, pad);
}

static GstStateChangeReturn
gst_tof_depth_change_state (GstElement * element, GstStateChange transition)
{
  GstTofDepth *self = GST_TOF_DEPTH (element);
  GstStateChangeReturn ret;

  if (transition == GST_STATE_CHANGE_READY_TO_PAUSED) {
    if (!load_models (self))
      return GST_STATE_CHANGE_FAILURE;
    gst_segment_init (&self->segment, GST_FORMAT_TIME);
    gst_flow_combiner_reset (self->combiner);
    self->hist_count = self->hist_pos = 0;
    g_clear_pointer (&self->hist, g_free);
  }
  ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);
  if (transition == GST_STATE_CHANGE_PAUSED_TO_READY) {
    for (guint k = 0; k < N_OUT; k++) {
      if (self->outs[k].pool) {
        gst_buffer_pool_set_active (self->outs[k].pool, FALSE);
        gst_clear_object (&self->outs[k].pool);
      }
      self->outs[k].need_caps = self->outs[k].need_stream_start = self->outs[k].need_segment = TRUE;
    }
    gst_clear_event (&self->stream_start);
    self->models_loaded = FALSE;
  }
  return ret;
}

/* ---------------------------------------------------------------- properties */

static void
gst_tof_depth_set_property (GObject * object, guint prop_id, const GValue * value,
    GParamSpec * pspec)
{
  GstTofDepth *self = GST_TOF_DEPTH (object);

  GST_OBJECT_LOCK (self);
  switch (prop_id) {
    case PROP_CALIBRATION:
      g_free (self->calibration);
      self->calibration = g_value_dup_string (value);
      self->models_loaded = FALSE;
      break;
    case PROP_MIN_AMPLITUDE:
      self->min_amplitude = g_value_get_double (value);
      break;
    case PROP_MIN_DISTANCE:
      self->min_distance = g_value_get_double (value);
      break;
    case PROP_MAX_DISTANCE:
      self->max_distance = g_value_get_double (value);
      break;
    case PROP_COLORMAP:
      self->colormap = g_value_get_enum (value);
      break;
    case PROP_AMPLITUDE_MAX:
      self->amplitude_max = g_value_get_double (value);
      break;
    case PROP_CONFIDENCE_SCALE:
      self->confidence_scale = g_value_get_double (value);
      break;
    case PROP_TEMPORAL_AVERAGE:
      self->temporal_average = g_value_get_uint (value);
      break;
    case PROP_DEPTH_TYPE:
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
  GST_OBJECT_UNLOCK (self);
}

static void
gst_tof_depth_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec)
{
  GstTofDepth *self = GST_TOF_DEPTH (object);

  GST_OBJECT_LOCK (self);
  switch (prop_id) {
    case PROP_CALIBRATION:
      g_value_set_string (value, self->calibration);
      break;
    case PROP_MIN_AMPLITUDE:
      g_value_set_double (value, self->min_amplitude);
      break;
    case PROP_MIN_DISTANCE:
      g_value_set_double (value, self->min_distance);
      break;
    case PROP_MAX_DISTANCE:
      g_value_set_double (value, self->max_distance);
      break;
    case PROP_COLORMAP:
      g_value_set_enum (value, self->colormap);
      break;
    case PROP_AMPLITUDE_MAX:
      g_value_set_double (value, self->amplitude_max);
      break;
    case PROP_CONFIDENCE_SCALE:
      g_value_set_double (value, self->confidence_scale);
      break;
    case PROP_TEMPORAL_AVERAGE:
      g_value_set_uint (value, self->temporal_average);
      break;
    case PROP_DEPTH_TYPE:
      g_value_set_enum (value, GST_TOF_DEPTH_RADIAL);
      break;
    case PROP_STATS:
      g_value_take_boxed (value, gst_structure_new ("stats",
              "groups", G_TYPE_UINT64, self->n_groups,
              "mean-process-us", G_TYPE_DOUBLE,
              self->n_groups ? self->proc_ns / 1000.0 / self->n_groups : 0.0, NULL));
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
  GST_OBJECT_UNLOCK (self);
}

static void
gst_tof_depth_finalize (GObject * object)
{
  GstTofDepth *self = GST_TOF_DEPTH (object);

  g_free (self->calibration);
  g_free (self->dist);
  g_free (self->amp);
  g_free (self->sum_i);
  g_free (self->sum_q);
  g_free (self->hist);
  gst_clear_event (&self->stream_start);
  gst_flow_combiner_free (self->combiner);
  G_OBJECT_CLASS (parent_class)->finalize (object);
}

static void
gst_tof_depth_class_init (GstTofDepthClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *element_class = GST_ELEMENT_CLASS (klass);
  const GParamFlags rw = G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS;

  gobject_class->set_property = gst_tof_depth_set_property;
  gobject_class->get_property = gst_tof_depth_get_property;
  gobject_class->finalize = gst_tof_depth_finalize;

  g_object_class_install_property (gobject_class, PROP_CALIBRATION,
      g_param_spec_string ("calibration", "Calibration",
          "Key file with the per-mode depth model (tools/tof-sweep.py CAL_OUT=*.conf); "
          "NULL = uncorrected, not metric", NULL, rw | GST_PARAM_MUTABLE_READY));
  g_object_class_install_property (gobject_class, PROP_MIN_AMPLITUDE,
      g_param_spec_double ("min-amplitude", "Minimum amplitude",
          "Pixels with a lower amplitude (counts) are invalid (0)", 0.0, 1024.0,
          DEFAULT_MIN_AMPLITUDE, rw | GST_PARAM_MUTABLE_PLAYING));
  g_object_class_install_property (gobject_class, PROP_MIN_DISTANCE,
      g_param_spec_double ("min-distance", "Minimum distance",
          "Colour-map start for visual depth (m)", 0.0, 10.0, DEFAULT_MIN_DISTANCE,
          rw | GST_PARAM_MUTABLE_PLAYING));
  g_object_class_install_property (gobject_class, PROP_MAX_DISTANCE,
      g_param_spec_double ("max-distance", "Maximum distance",
          "Colour-map end for visual depth (m); 0 = the mode's unambiguous range", 0.0, 10.0,
          DEFAULT_MAX_DISTANCE, rw | GST_PARAM_MUTABLE_PLAYING));
  g_object_class_install_property (gobject_class, PROP_COLORMAP,
      g_param_spec_enum ("colormap", "Colour map", "Colour map of visual depth",
          gst_tof_colormap_get_type (), GST_TOF_COLORMAP_TURBO, rw | GST_PARAM_MUTABLE_PLAYING));
  g_object_class_install_property (gobject_class, PROP_AMPLITUDE_MAX,
      g_param_spec_double ("amplitude-max", "Amplitude maximum",
          "Amplitude (counts) shown as 255 on a GRAY8 amplitude output", 1.0, 2048.0,
          DEFAULT_AMPLITUDE_MAX, rw | GST_PARAM_MUTABLE_PLAYING));
  g_object_class_install_property (gobject_class, PROP_CONFIDENCE_SCALE,
      g_param_spec_double ("confidence-scale", "Confidence scale",
          "Amplitude (counts) at which confidence saturates; confidence is proportional to "
          "amplitude below it (depth noise ~ 1/amplitude)", 1.0, 2048.0,
          DEFAULT_CONFIDENCE_SCALE, rw | GST_PARAM_MUTABLE_PLAYING));
  g_object_class_install_property (gobject_class, PROP_TEMPORAL_AVERAGE,
      g_param_spec_uint ("temporal-average", "Temporal average",
          "Average I/Q over a sliding window of N groups of the same mode (software substitute "
          "for exposure, DEC-0017)", 1, MAX_AVERAGE, DEFAULT_TEMPORAL_AVERAGE,
          rw | GST_PARAM_MUTABLE_PLAYING));
  g_object_class_install_property (gobject_class, PROP_DEPTH_TYPE,
      g_param_spec_enum ("depth-type", "Depth type", "Distance definition of the depth output",
          gst_tof_depth_type_get_type (), GST_TOF_DEPTH_RADIAL, rw | GST_PARAM_MUTABLE_READY));
  g_object_class_install_property (gobject_class, PROP_STATS,
      g_param_spec_boxed ("stats", "Statistics", "groups, mean-process-us",
          GST_TYPE_STRUCTURE, G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

  gst_element_class_add_static_pad_template (element_class, &sink_template);
  gst_element_class_add_static_pad_template (element_class, &depth_template);
  gst_element_class_add_static_pad_template (element_class, &amplitude_template);
  gst_element_class_add_static_pad_template (element_class, &confidence_template);
  gst_element_class_set_static_metadata (element_class, "Arducam ToF depth",
      "Filter/Converter/Video", "Calibrated depth, amplitude and confidence from groups of "
      "4 DCS frames (qtitofsrc)", "arducam-tof-ventuno project");

  element_class->request_new_pad = GST_DEBUG_FUNCPTR (gst_tof_depth_request_new_pad);
  element_class->release_pad = GST_DEBUG_FUNCPTR (gst_tof_depth_release_pad);
  element_class->change_state = GST_DEBUG_FUNCPTR (gst_tof_depth_change_state);

  GST_DEBUG_CATEGORY_INIT (gst_tof_depth_debug, "qtitofdepth", 0, "Arducam ToF depth");
}

static void
gst_tof_depth_init (GstTofDepth * self)
{
  GstPad *src;

  self->min_amplitude = DEFAULT_MIN_AMPLITUDE;
  self->min_distance = DEFAULT_MIN_DISTANCE;
  self->max_distance = DEFAULT_MAX_DISTANCE;
  self->colormap = GST_TOF_COLORMAP_TURBO;
  self->amplitude_max = DEFAULT_AMPLITUDE_MAX;
  self->confidence_scale = DEFAULT_CONFIDENCE_SCALE;
  self->temporal_average = DEFAULT_TEMPORAL_AVERAGE;
  self->fps_n = 30;
  self->fps_d = 1;
  self->hist_mode = -1;
  self->dist = g_new0 (gfloat, N_PIX);
  self->amp = g_new0 (gfloat, N_PIX);
  self->sum_i = g_new0 (gfloat, N_PIX);
  self->sum_q = g_new0 (gfloat, N_PIX);
  build_turbo (self->turbo);
  gst_segment_init (&self->segment, GST_FORMAT_TIME);
  self->combiner = gst_flow_combiner_new ();

  self->sinkpad = gst_pad_new_from_static_template (&sink_template, "sink");
  gst_pad_set_chain_function (self->sinkpad, GST_DEBUG_FUNCPTR (gst_tof_depth_chain));
  gst_pad_set_event_function (self->sinkpad, GST_DEBUG_FUNCPTR (gst_tof_depth_sink_event));
  gst_element_add_pad (GST_ELEMENT (self), self->sinkpad);

  src = gst_pad_new_from_static_template (&depth_template, "src");
  setup_src_pad (self, &self->outs[OUT_DEPTH], src);
  gst_element_add_pad (GST_ELEMENT (self), src);
  gst_flow_combiner_add_pad (self->combiner, src);
}
