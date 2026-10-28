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
 * depth-type=z (arducam-tof-ventuno RUN-0085): z = r / |(x, y, 1)| with (x, y) the undistorted normalised
 * coordinates of the pixel, from a lens model (key-file group [lens]: f, cx, cy, k1, k2 - the model of
 * tools/tof-lens.py) in the lens= file or the calibration file; a per-pixel table built at load time, applied
 * after the flying-pixel pass.
 *
 * wrap-reflectivity (arducam-tof-ventuno RUN-0090/RUN-0091): one modulation frequency per mode, so a surface beyond R
 * reads r = d - R. A target one range further gives at most rho K / (r + R)^2 (K = wrap_k of the calibration's mode
 * group, the reference target's amplitude x m^2); a valid pixel whose amplitude is not above that bound plus
 * 3 wrap_sigma / sqrt (groups averaged) could be wrapped and becomes invalid, before the flying-pixel pass.
 *
 * unwrap (arducam-tof-ventuno RUN-0105/RUN-0106): near mode reads a surface at d >= R_near as d - R_near. Every far group
 * (qtitofsrc far-reference-interval) is a reference: per pixel k = argmin |r_B + k R_near - r_F|, k in {0, 1}, with r_B
 * the latest valid near reading since the previous reference; D = r_B + k R_near if that residual <= unwrap-delta, else
 * the previous D stays. After the first reference a valid near pixel with a D is output at r + k' R_near when
 * |r + k' R_near - D| <= unwrap-tolerance (k' the argmin), else it becomes invalid. Far groups are output unchanged, near
 * groups before the first reference too; on the measured distances, before the other passes; once a reference exists
 * the wrap test is not applied to near groups (unwrap supersedes it). One loop for both paths (no NEON variant).
 *
 * References (arducam-tof-ventuno RUN-0110): a group flagged GST_TOF_FLAG_REFERENCE is computed from its own I/Q and leaves
 * the temporal-average window as it is, so the near groups on both sides of it average across it (an unflagged mode change
 * still restarts the window). output-references=false leaves such a group out of every output - unwrap still takes it -
 * and pushes a GAP event at its PTS instead.
 *
 * cpu-affinity / util-clamp-min (arducam-tof-ventuno RUN-0098): a scheduling hint for the streaming thread that calls the
 * chain function (without a queue in front, qtitofsrc's), instead of taskset on the whole process. Applied with
 * sched_setaffinity / sched_setattr (SCHED_FLAG_UTIL_CLAMP_MIN, unprivileged) when the first group arrives on a thread; that
 * thread's previous CPU set and uclamp.min are restored at PAUSED -> READY or when another thread takes over. Unset = no call.
 *
 * Pads: "src" (always) depth, GRAY16_LE millimetres or a colour-mapped view; "amplitude" and
 * "confidence" (request); "pointcloud" (request, RUN-0086): the QIM SDK tensor format
 * (neural-network/tensors, FLOAT32, 1 x 180 x 240 x 3 + GstMLTensorMeta), per pixel X, Y, Z in metres
 * (x right, y down, z forward) from the lens model, NaN = invalid. Outputs not linked are not computed. Invalid pixels (amplitude below
 * min-amplitude, or a DCS sample at the 11-bit rails; optionally possible range wraps and flying pixels) are 0.
 * Output buffers come from the downstream image pool or from a QIM image pool with the QTI DMA-buf allocator
 * (videotransform.c pattern).
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE             /* sched_setaffinity, CPU_SET, gettid */
#endif

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "tofdepth.h"
#include "tofdcs.h"

#include <errno.h>
#include <math.h>
#include <sched.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#if defined(__aarch64__) && defined(__ARM_NEON) && G_BYTE_ORDER == G_LITTLE_ENDIAN
#include <arm_neon.h>
#define TOF_HAVE_NEON 1
#endif

#include <gst/base/gstflowcombiner.h>
#include <gst/video/video.h>
#include <gst/allocators/gstqtiallocator.h>
#include <gst/video/gstimagepool.h>
#include <gst/ml/ml-info.h>
#include <gst/ml/gstmlmeta.h>

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
#define DEFAULT_FLYING_PIXEL_THRESHOLD 0.0
#define DEFAULT_WRAP_REFLECTIVITY 0.0
#define WRAP_NOISE_Z             3.0     /* noise allowance in sigma (RUN-0090) */
#define DEFAULT_UNWRAP_DELTA     0.5     /* m: R_near / 4 (RUN-0105) */
#define DEFAULT_UNWRAP_TOLERANCE 0.15    /* m: 0.075 R_near (RUN-0105) */
#define DEFAULT_OUTPUT_REFERENCES TRUE
#define DEFAULT_PROCESSING_LATENCY -1
#define DEFAULT_UTIL_CLAMP_MIN   -1
/* automatic processing latency: live p99 after the source push, NEON build, unpinned (arducam-tof-ventuno
 * RUN-0064): ~6 ms with one output, ~9.3 ms for the third of three (max 11.2) */
#define AUTO_LATENCY_BASE        (4 * GST_MSECOND)
#define AUTO_LATENCY_PER_OUTPUT  (2500 * GST_USECOND)

#define DEPTH_FORMATS "{ GRAY16_LE, RGBA, BGRA, RGB, BGR, GRAY8 }"
/* RGB(A) variants replicate the grey value: qtivcomposer (GLES) shows GRAY8 as red only (RUN-0049) */
#define AMPLITUDE_FORMATS "{ GRAY16_LE, GRAY8, RGBA, BGRA, RGB, BGR }"
#define CONFIDENCE_FORMATS "{ GRAY8, GRAY16_LE, RGBA, BGRA, RGB, BGR }"
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
  GST_TOF_DEPTH_Z,
} GstTofDepthType;

enum
{
  OUT_DEPTH,
  OUT_AMPLITUDE,
  OUT_CONFIDENCE,
  OUT_POINTCLOUD,
  N_OUT
};

#define POINTCLOUD_SIZE  (N_PIX * 3 * sizeof (gfloat))

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
  PROP_FLYING_PIXEL_THRESHOLD,
  PROP_WRAP_REFLECTIVITY,
  PROP_DEPTH_TYPE,
  PROP_LENS,
  PROP_PROCESSING_LATENCY,
  PROP_CPU_AFFINITY,
  PROP_UTIL_CLAMP_MIN,
  PROP_UNWRAP,
  PROP_UNWRAP_DELTA,
  PROP_UNWRAP_TOLERANCE,
  PROP_OUTPUT_REFERENCES,
  PROP_STATS,
};

/* struct sched_attr of the kernel ABI (SCHED_ATTR_SIZE_VER1, include/uapi/linux/sched/types.h): glibc before 2.41 has no
 * sched_setattr wrapper */
typedef struct
{
  guint32 size;
  guint32 sched_policy;
  guint64 sched_flags;
  gint32 sched_nice;
  guint32 sched_priority;
  guint64 sched_runtime, sched_deadline, sched_period;
  guint32 sched_util_min, sched_util_max;
} TofSchedAttr;

#define TOF_SCHED_FLAG_KEEP_POLICY    0x08
#define TOF_SCHED_FLAG_KEEP_PARAMS    0x10
#define TOF_SCHED_FLAG_UTIL_CLAMP_MIN 0x20
#define TOF_UTIL_CLAMP_RESET          G_MAXUINT32       /* (u32) -1: back to the default request */

typedef struct
{
  gdouble r;
  gdouble phi_ref;
  gdouble c0, c1;
  gdouble a[MAX_HARMONICS], b[MAX_HARMONICS];   /* terms of 4 (h + 1) theta */
  gfloat lut[LUT_SIZE + 1];                     /* unwrapped distance (m) over u */
  gdouble wrap_k, wrap_sigma;                   /* wrap reference: amplitude x m^2, noise (counts); 0 = none */
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
  gchar *calibration_used;      /* file in use, "" = uncalibrated (stats) */
  gdouble min_amplitude;
  gdouble min_distance;
  gdouble max_distance;
  GstTofColormap colormap;
  gdouble amplitude_max;
  gdouble confidence_scale;
  guint temporal_average;
  gdouble flying_pixel_threshold;  /* m, 0 = off */
  gdouble wrap_reflectivity;    /* relative to the calibration's wrap_k, 0 = off */
  GstTofDepthType depth_type;
  gchar *lens;                  /* key file with [lens], NULL = the calibration's [lens] group if any */
  gchar *lens_used;             /* file in use, NULL = no lens (stats) */
  gint64 processing_latency;    /* ns, -1 = automatic */
  gchar *cpu_affinity;          /* CPU list for the streaming thread, NULL or "" = leave it alone */
  gint util_clamp_min;          /* uclamp.min for the streaming thread, -1 = leave it alone */
  gboolean unwrap;              /* far groups as references for near groups */
  gdouble unwrap_delta;         /* m, reference residual limit */
  gdouble unwrap_tolerance;     /* m, carry residual limit */
  gboolean output_references;   /* FALSE = groups flagged as references are not output */

  /* thread hint: set at READY -> PAUSED, applied by the streaming thread, undone at PAUSED -> READY */
  gboolean hint_cpus;           /* hint_set is to be applied */
  cpu_set_t hint_set;
  gint hint_util_min;
  GThread *hint_thread;         /* thread the hint was applied to, NULL = none */
  pid_t hint_tid;
  gboolean saved_cpus_valid;
  cpu_set_t saved_set;          /* the thread's CPU set before the hint */
  gint saved_util_min;          /* the thread's uclamp.min before the hint, -1 = not changed */
  gchar *thread_cpus;           /* read back after applying (stats, object lock), NULL = no hint */
  gint thread_util_min;         /* read back after applying (stats, object lock), -1 = no hint */

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
  guint8 *flying;               /* flying-pixel mask of the current group */
  gfloat *zscale;               /* z / r per pixel, NULL = no lens loaded */
  gfloat *rayx, *rayy;          /* undistorted normalised x, y per pixel (X = z x, Y = z y) */
  gboolean dist_is_z;           /* dist of the current group already scaled to z */
  guint hist_count, hist_pos, hist_n;
  gint hist_mode;
  guint avg_count;              /* groups averaged into the current one */
  gfloat *uw_ref;               /* unwrap: reference distance D per pixel (m), 0 = none */
  gfloat *uw_last;              /* unwrap: latest valid near reading since the previous reference, 0 = none */
  gboolean uw_seen;             /* unwrap: a near group was seen */
  gboolean uw_have;             /* unwrap: a reference was taken */

  /* stats (object lock) */
  guint64 n_groups;
  guint64 proc_ns;
  guint64 n_flying;
  guint64 n_wrap;
  guint64 n_uw_refs;
  guint64 n_unwrapped;
  guint64 n_undecided;
  guint64 n_refs_dropped;
};

#define gst_tof_depth_parent_class parent_class
G_DEFINE_TYPE (GstTofDepth, gst_tof_depth, GST_TYPE_ELEMENT);

#ifdef TOF_HAVE_NEON
/* NEON path (RUN-0064) unless GST_TOF_DEPTH_SCALAR is set: the scalar code is the reference */
static gboolean use_simd = FALSE;
#endif

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
    { GST_TOF_DEPTH_RADIAL, "Radial distance along each pixel's ray (as calibrated)", "radial" },
    { GST_TOF_DEPTH_Z, "Distance along the optical axis (needs a lens model: lens= or the calibration's "
          "[lens] group)", "z" },
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
  if (g_key_file_has_key (kf, group, "wrap_k", NULL))
    m->wrap_k = g_key_file_get_double (kf, group, "wrap_k", NULL);
  if (g_key_file_has_key (kf, group, "wrap_sigma", NULL))
    m->wrap_sigma = g_key_file_get_double (kf, group, "wrap_sigma", NULL);
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
load_calibration (GstTofDepth * self)
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
    /* unset: the installed calibration (DEC-0026) */
    const gchar *env = g_getenv ("GST_TOF_CALIBRATION");

    path = g_strdup (env && *env ? env : TOF_DEFAULT_CALIBRATION);
    if (!g_file_test (path, G_FILE_TEST_IS_REGULAR)) {
      GST_ELEMENT_ERROR (self, RESOURCE, NOT_FOUND,
          ("No depth calibration: %s does not exist.", path),
          ("Set calibration= to a key file written by tools/tof-sweep.py (CAL_OUT=*.conf), or "
              "calibration=\"\" for uncalibrated depth (not metric); GST_TOF_CALIBRATION sets the default."));
      GST_OBJECT_LOCK (self);
      g_clear_pointer (&self->calibration_used, g_free);
      GST_OBJECT_UNLOCK (self);
      g_free (path);
      return FALSE;
    }
    GST_INFO_OBJECT (self, "calibration unset: using %s", path);
  }
  if (!*path) {
    GST_INFO_OBJECT (self, "calibration=\"\": uncalibrated depth, not metric");
    GST_OBJECT_LOCK (self);
    g_free (self->calibration_used);
    self->calibration_used = path;
    GST_OBJECT_UNLOCK (self);
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
  GST_OBJECT_LOCK (self);
  g_free (self->calibration_used);
  self->calibration_used = ok ? path : NULL;
  GST_OBJECT_UNLOCK (self);
  if (!ok)
    g_free (path);
  return ok;
}

/* Lens model [lens]: f, cx, cy (pixels; u, v = 0-based pixel centres), k1, k2 of the inverse radial model
 * x = xd (1 + k1 rd^2 + k2 rd^4), xd = (u - cx) / f (likewise y); optional width/height must be the sensor's.
 * z / r = 1 / |(x, y, 1)| per pixel. */
static gboolean
lens_parse (GstTofDepth * self, GKeyFile * kf, const gchar * path, gfloat * zscale, gfloat * rayx,
    gfloat * rayy)
{
  GError *err = NULL;
  gdouble f, cx, cy, k1, k2 = 0.0;

  f = g_key_file_get_double (kf, "lens", "f", &err);
  if (!err)
    cx = g_key_file_get_double (kf, "lens", "cx", &err);
  if (!err)
    cy = g_key_file_get_double (kf, "lens", "cy", &err);
  if (!err)
    k1 = g_key_file_get_double (kf, "lens", "k1", &err);
  if (!err && g_key_file_has_key (kf, "lens", "k2", NULL))
    k2 = g_key_file_get_double (kf, "lens", "k2", &err);
  if (err) {
    GST_ERROR_OBJECT (self, "lens %s: %s", path, err->message);
    g_error_free (err);
    return FALSE;
  }
  if (!(f > 0.0) || (g_key_file_has_key (kf, "lens", "width", NULL) &&
          g_key_file_get_integer (kf, "lens", "width", NULL) != GST_TOF_WIDTH) ||
      (g_key_file_has_key (kf, "lens", "height", NULL) &&
          g_key_file_get_integer (kf, "lens", "height", NULL) != GST_TOF_HEIGHT)) {
    GST_ERROR_OBJECT (self, "lens %s: f must be > 0 and width x height %ux%u", path, GST_TOF_WIDTH,
        GST_TOF_HEIGHT);
    return FALSE;
  }
  for (guint v = 0; v < GST_TOF_HEIGHT; v++) {
    for (guint u = 0; u < GST_TOF_WIDTH; u++) {
      gdouble xd = (u - cx) / f, yd = (v - cy) / f, r2 = xd * xd + yd * yd;
      gdouble k = 1.0 + k1 * r2 + k2 * r2 * r2, x = xd * k, y = yd * k;

      zscale[v * GST_TOF_WIDTH + u] = (gfloat) (1.0 / sqrt (1.0 + x * x + y * y));
      rayx[v * GST_TOF_WIDTH + u] = (gfloat) x;
      rayy[v * GST_TOF_WIDTH + u] = (gfloat) y;
    }
  }
  GST_INFO_OBJECT (self, "lens %s: f %.3f, cx %.2f, cy %.2f, k1 %+.5f, k2 %+.5f; corner z/r %.5f", path, f, cx,
      cy, k1, k2, zscale[0]);
  return TRUE;
}

/* the lens of lens= (errors if unusable), else the [lens] group of the calibration file cal_path (if any);
 * depth-type=z without a lens fails */
static gboolean
load_lens (GstTofDepth * self, const gchar * cal_path)
{
  GstTofDepthType type;
  gchar *path;
  gboolean explicit, have = FALSE;

  GST_OBJECT_LOCK (self);
  path = g_strdup (self->lens);
  type = self->depth_type;
  GST_OBJECT_UNLOCK (self);
  explicit = path && *path;
  if (!explicit) {
    g_free (path);
    path = cal_path && *cal_path ? g_strdup (cal_path) : NULL;
  }
  g_clear_pointer (&self->zscale, g_free);
  g_clear_pointer (&self->rayx, g_free);
  g_clear_pointer (&self->rayy, g_free);
  if (path) {
    GKeyFile *kf = g_key_file_new ();
    GError *err = NULL;

    if (!g_key_file_load_from_file (kf, path, G_KEY_FILE_NONE, &err)) {
      if (explicit || type == GST_TOF_DEPTH_Z) {
        GST_ELEMENT_ERROR (self, RESOURCE, OPEN_READ, ("Cannot read lens %s.", path), ("%s", err->message));
        g_error_free (err);
        g_key_file_free (kf);
        g_free (path);
        return FALSE;
      }
      g_clear_error (&err);
    } else if (g_key_file_has_group (kf, "lens")) {
      self->zscale = g_new (gfloat, N_PIX);
      self->rayx = g_new (gfloat, N_PIX);
      self->rayy = g_new (gfloat, N_PIX);
      have = lens_parse (self, kf, path, self->zscale, self->rayx, self->rayy);
      if (!have) {
        g_clear_pointer (&self->zscale, g_free);
        g_clear_pointer (&self->rayx, g_free);
        g_clear_pointer (&self->rayy, g_free);
        if (explicit || type == GST_TOF_DEPTH_Z) {
          GST_ELEMENT_ERROR (self, RESOURCE, SETTINGS, ("Invalid lens model in %s.", path),
              ("[lens] needs f > 0, cx, cy, k1 (k2 optional); width/height, if given, %ux%u.", GST_TOF_WIDTH,
                  GST_TOF_HEIGHT));
          g_key_file_free (kf);
          g_free (path);
          return FALSE;
        }
        GST_WARNING_OBJECT (self, "ignoring the invalid [lens] group of %s (depth-type=radial)", path);
      }
    } else if (explicit) {
      GST_ELEMENT_ERROR (self, RESOURCE, SETTINGS, ("%s has no [lens] group.", path), (NULL));
      g_key_file_free (kf);
      g_free (path);
      return FALSE;
    }
    g_key_file_free (kf);
  }
  if (!have && type == GST_TOF_DEPTH_Z) {
    GST_ELEMENT_ERROR (self, RESOURCE, SETTINGS, ("depth-type=z needs a lens model."),
        ("Set lens= to a key file with a [lens] group (f, cx, cy, k1, optional k2; tools/tof-lens.py writes one), "
            "or add that group to the calibration file."));
    g_free (path);
    return FALSE;
  }
  GST_OBJECT_LOCK (self);
  g_free (self->lens_used);
  self->lens_used = have ? path : NULL;
  GST_OBJECT_UNLOCK (self);
  if (!have)
    g_free (path);
  return TRUE;
}

static gboolean
load_models (GstTofDepth * self)
{
  gchar *cal;
  gboolean ok;

  if (!load_calibration (self))
    return FALSE;
  GST_OBJECT_LOCK (self);
  cal = g_strdup (self->calibration_used);
  GST_OBJECT_UNLOCK (self);
  ok = load_lens (self, cal);
  g_free (cal);
  if (!ok)
    self->models_loaded = FALSE;
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

/* atan (t), t in [0, 1] = t (1 + c2 t^2 + ... + c16 t^16): Abramowitz & Stegun 4.4.49, |error| <= 2e-8 rad,
 * i.e. < 0.1 um of depth at R = 4 m; the float rounding (~1e-7 rad) dominates */
static const gfloat atan_c[8] = { -0.3333314528f, 0.1999355085f, -0.1420889944f, 0.1065626393f,
  -0.0752896400f, 0.0429096138f, -0.0161657367f, 0.0028662257f
};

/* atan2_f32x4 () for one pixel: the same polynomial, operations and order (fmaf where it fuses), so the scalar loop
 * of compute () is bit-identical to the NEON path. libm atan2f differs from the polynomial by an ulp in about a third
 * of the pixels, which moved 91 of 68.7 M depth values by 1 mm (arducam-tof-ventuno RUN-0106, RUN-0108). */
static inline gfloat
atan2_poly (gfloat y, gfloat x)
{
  gfloat ax = fabsf (x), ay = fabsf (y), mx = MAX (ax, ay), mn = MIN (ax, ay);
  gfloat t = mn / (mx == 0.0f ? 1.0f : mx), z = t * t, p = atan_c[7], r;

  for (gint k = 6; k >= 0; k--)
    p = fmaf (p, z, atan_c[k]);
  r = t * fmaf (p, z, 1.0f);
  r = ay > ax ? (gfloat) G_PI_2 - r : r;
  r = x < 0.0f ? (gfloat) G_PI - r : r;
  return y < 0.0f ? -r : r;
}

#ifdef TOF_HAVE_NEON
static inline float32x4_t
atan2_f32x4 (float32x4_t y, float32x4_t x)
{
  const float32x4_t zero = vdupq_n_f32 (0.0f);
  float32x4_t ax = vabsq_f32 (x), ay = vabsq_f32 (y);
  float32x4_t mx = vmaxq_f32 (ax, ay), mn = vminq_f32 (ax, ay);
  float32x4_t t = vdivq_f32 (mn, vbslq_f32 (vceqq_f32 (mx, zero), vdupq_n_f32 (1.0f), mx));
  float32x4_t z = vmulq_f32 (t, t), p = vdupq_n_f32 (atan_c[7]), r;

  for (gint k = 6; k >= 0; k--)
    p = vfmaq_f32 (vdupq_n_f32 (atan_c[k]), p, z);
  r = vmulq_f32 (t, vfmaq_f32 (vdupq_n_f32 (1.0f), p, z));
  /* octants: |y| > |x| -> pi/2 - r; x < 0 -> pi - r; y < 0 -> -r (atan2 (+0, x < 0) = +pi as libm) */
  r = vbslq_f32 (vcgtq_f32 (ay, ax), vsubq_f32 (vdupq_n_f32 ((gfloat) G_PI_2), r), r);
  r = vbslq_f32 (vcltq_f32 (x, zero), vsubq_f32 (vdupq_n_f32 ((gfloat) G_PI), r), r);
  return vbslq_f32 (vcltq_f32 (y, zero), vnegq_f32 (r), r);
}

/* the per-pixel loop of compute (), 4 pixels per step; same operations and order as the scalar loop, bit-identical (RUN-0108) */
static void
compute_neon (GstTofDepth * self, const gint16 * f0, const gint16 * f1, const gint16 * f2,
    const gint16 * f3, const TofModel * m, gfloat min_amp, guint navg, gboolean want_depth)
{
  const float32x4_t zero = vdupq_n_f32 (0.0f), two_pi = vdupq_n_f32 ((gfloat) TWO_PI);
  const float32x4_t ref = vdupq_n_f32 ((gfloat) m->phi_ref), r = vdupq_n_f32 ((gfloat) m->r);
  const float32x4_t scale = vdupq_n_f32 (LUT_SIZE / (gfloat) TWO_PI);
  const float32x4_t vmin_amp = vdupq_n_f32 (min_amp), count = vdupq_n_f32 ((gfloat) self->hist_count);
  const int32x4_t hi = vdupq_n_s32 (1023), lo = vdupq_n_s32 (-1024);
  const uint32x4_t kmax = vdupq_n_u32 (LUT_SIZE - 1);
  const gfloat *lut = m->lut;

  for (guint p = 0; p < N_PIX; p += 4) {
    int32x4_t s0 = vmovl_s16 (vld1_s16 (f0 + p)), s1 = vmovl_s16 (vld1_s16 (f1 + p));
    int32x4_t s2 = vmovl_s16 (vld1_s16 (f2 + p)), s3 = vmovl_s16 (vld1_s16 (f3 + p));
    uint32x4_t rail = vorrq_u32 (vorrq_u32 (vcgeq_s32 (s0, hi), vcleq_s32 (s0, lo)),
        vorrq_u32 (vcgeq_s32 (s1, hi), vcleq_s32 (s1, lo)));
    float32x4_t i, q, a, d;
    uint32x4_t invalid;

    rail = vorrq_u32 (rail, vorrq_u32 (vorrq_u32 (vcgeq_s32 (s2, hi), vcleq_s32 (s2, lo)),
            vorrq_u32 (vcgeq_s32 (s3, hi), vcleq_s32 (s3, lo))));
    if (navg > 1) {
      i = vdivq_f32 (vld1q_f32 (self->sum_i + p), count);
      q = vdivq_f32 (vld1q_f32 (self->sum_q + p), count);
    } else {
      i = vcvtq_f32_s32 (vsubq_s32 (s0, s2));
      q = vcvtq_f32_s32 (vsubq_s32 (s1, s3));
    }
    a = vmulq_f32 (vsqrtq_f32 (vfmaq_f32 (vmulq_f32 (i, i), q, q)), vdupq_n_f32 (0.5f));
    vst1q_f32 (self->amp + p, a);
    invalid = vorrq_u32 (rail, vcltq_f32 (a, vmin_amp));
    if (want_depth) {
      float32x4_t u = vsubq_f32 (ref, atan2_f32x4 (q, i)), x, l0, l1;
      guint32 k[4];
      uint32x4_t kv;

      u = vfmsq_f32 (u, two_pi, vrndmq_f32 (vdivq_f32 (u, two_pi)));
      x = vmulq_f32 (u, scale);
      kv = vminq_u32 (vcvtq_u32_f32 (x), kmax);
      vst1q_u32 (k, kv);
      l0 = (float32x4_t) { lut[k[0]], lut[k[1]], lut[k[2]], lut[k[3]] };
      l1 = (float32x4_t) { lut[k[0] + 1], lut[k[1] + 1], lut[k[2] + 1], lut[k[3] + 1] };
      d = vfmaq_f32 (l0, vsubq_f32 (x, vcvtq_f32_u32 (kv)), vsubq_f32 (l1, l0));
      d = vfmsq_f32 (d, r, vrndmq_f32 (vdivq_f32 (d, r)));
      d = vbslq_f32 (vcgtq_f32 (d, zero), d, vdupq_n_f32 (1e-4f));
    } else {
      d = vdupq_n_f32 (1.0f);   /* valid marker only */
    }
    vst1q_f32 (self->dist + p, vbslq_f32 (invalid, zero, d));
  }
}
#endif

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
  /* a reference (qtitofsrc far-reference-interval) is computed on its own and leaves the window as it is (RUN-0110) */
  if (GST_READ_UINT16_LE (data + GST_TOF_OFF_FLAGS) & GST_TOF_FLAG_REFERENCE)
    navg = 1;

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
  self->avg_count = navg > 1 ? self->hist_count : 1;

#ifdef TOF_HAVE_NEON
  if (use_simd) {
    compute_neon (self, f0, f1, f2, f3, m, min_amp, navg, want_depth);
    return;
  }
#endif
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
    /* the fused steps are written as fmaf, as vfmaq/vfmsq in compute_neon (), not left to fp contraction */
    a = sqrtf (fmaf (q, q, i * i)) * 0.5f;
    self->amp[p] = a;
    if (rail || a < min_amp) {
      self->dist[p] = 0.0f;
      continue;
    }
    if (want_depth) {
      gfloat u = ref - atan2_poly (q, i), x, d;
      guint k;

      u = fmaf (-(gfloat) TWO_PI, floorf (u / (gfloat) TWO_PI), u);
      x = u * (LUT_SIZE / (gfloat) TWO_PI);
      k = MIN ((guint) x, LUT_SIZE - 1);
      d = fmaf (x - k, m->lut[k + 1] - m->lut[k], m->lut[k]);
      d = fmaf (-r, floorf (d / r), d);
      self->dist[p] = d > 0.0f ? d : 1e-4f;
    } else {
      self->dist[p] = 1.0f;     /* valid marker only */
    }
  }
}

/* Range wrap: a valid pixel whose amplitude a target rho times as bright as the wrap reference could give from one
 * range further (r + R) - within the noise allowance - becomes invalid. Radial distance, before the flying pixels.
 * A <= rho K / (r + R)^2 + noise is tested as (A - noise) (r + R)^2 <= rho K (no division); the NEON path does the
 * same float operations in the same order, 4 pixels per step (bit-identical; the scalar loop is latency-bound on the
 * in-order A55, RUN-0091). */
static guint
remove_wrapped (GstTofDepth * self, const TofModel * m, gfloat rho)
{
  const gfloat k = rho * (gfloat) m->wrap_k, r = (gfloat) m->r;
  const gfloat noise = (gfloat) (WRAP_NOISE_Z * m->wrap_sigma / sqrt ((gdouble) self->avg_count));
  gfloat *dist = self->dist;
  const gfloat *amp = self->amp;
  guint n = 0;

#ifdef TOF_HAVE_NEON
  if (use_simd) {
    const float32x4_t vk = vdupq_n_f32 (k), vr = vdupq_n_f32 (r), vn = vdupq_n_f32 (noise);
    const float32x4_t zero = vdupq_n_f32 (0.0f);
    uint32x4_t cnt = vdupq_n_u32 (0);

    for (guint p = 0; p < N_PIX; p += 4) {
      float32x4_t d = vld1q_f32 (dist + p), s = vaddq_f32 (d, vr);
      uint32x4_t w = vandq_u32 (vcgtq_f32 (d, zero),
          vcleq_f32 (vmulq_f32 (vsubq_f32 (vld1q_f32 (amp + p), vn), vmulq_f32 (s, s)), vk));

      vst1q_f32 (dist + p, vbslq_f32 (w, zero, d));
      cnt = vsubq_u32 (cnt, w);         /* a set lane is all ones = -1 */
    }
    return vaddvq_u32 (cnt);
  }
#endif
  for (guint p = 0; p < N_PIX; p++) {
    gfloat d = dist[p], s = d + r;
    guint w = (d > 0.0f) & ((amp[p] - noise) * (s * s) <= k);

    dist[p] = w ? 0.0f : d;
    n += w;
  }
  return n;
}

static void
unwrap_reset (GstTofDepth * self)
{
  memset (self->uw_ref, 0, N_PIX * sizeof (gfloat));
  memset (self->uw_last, 0, N_PIX * sizeof (gfloat));
  self->uw_seen = self->uw_have = FALSE;
}

/* Unwrap (see the top of the file): a far group updates the references, a near group after the first reference is
 * extended by k' R_near or set invalid. The NEON path does the same float operations in the same order, 4 pixels per
 * step (bit-identical; the scalar loop costs +0.6 ms per group on the in-order A55, RUN-0106). Returns TRUE if the
 * group was unwrapped; counts the pixels given + R_near and those removed. */
static gboolean
unwrap_group (GstTofDepth * self, guint mode, gfloat delta, gfloat tol, guint * n_unw, guint * n_und)
{
  const gfloat rn = (gfloat) self->models[GST_TOF_RANGE_NEAR].r;
  gfloat *dist = self->dist, *ref = self->uw_ref, *last = self->uw_last;
  guint unw = 0, und = 0;

  if (mode == GST_TOF_RANGE_FAR) {
    if (!self->uw_seen)
      return FALSE;
#ifdef TOF_HAVE_NEON
    if (use_simd) {
      const float32x4_t vrn = vdupq_n_f32 (rn), vdelta = vdupq_n_f32 (delta), zero = vdupq_n_f32 (0.0f);

      for (guint p = 0; p < N_PIX; p += 4) {
        float32x4_t b = vld1q_f32 (last + p), f = vld1q_f32 (dist + p), br = vaddq_f32 (b, vrn);
        float32x4_t e0 = vabsq_f32 (vsubq_f32 (b, f)), e1 = vabsq_f32 (vsubq_f32 (br, f));
        uint32x4_t ok = vandq_u32 (vandq_u32 (vcgtq_f32 (b, zero), vcgtq_f32 (f, zero)),
            vcleq_f32 (vminq_f32 (e0, e1), vdelta));

        vst1q_f32 (ref + p, vbslq_f32 (ok, vbslq_f32 (vcltq_f32 (e1, e0), br, b), vld1q_f32 (ref + p)));
        vst1q_f32 (last + p, zero);
      }
      goto far_done;
    }
#endif
    for (guint p = 0; p < N_PIX; p++) {
      gfloat b = last[p], f = dist[p], br = b + rn;
      gfloat e0 = fabsf (b - f), e1 = fabsf (br - f);
      gboolean k = e1 < e0, ok = (b > 0.0f) & (f > 0.0f) & (MIN (e0, e1) <= delta);

      ref[p] = ok ? (k ? br : b) : ref[p];
      last[p] = 0.0f;
    }
#ifdef TOF_HAVE_NEON
  far_done:
#endif
    self->uw_have = TRUE;
    GST_OBJECT_LOCK (self);
    self->n_uw_refs++;
    GST_OBJECT_UNLOCK (self);
    return FALSE;
  }
  self->uw_seen = TRUE;
  if (!self->uw_have) {
    for (guint p = 0; p < N_PIX; p++)
      last[p] = dist[p] > 0.0f ? dist[p] : last[p];
    return FALSE;
  }
#ifdef TOF_HAVE_NEON
  if (use_simd) {
    const float32x4_t vrn = vdupq_n_f32 (rn), vtol = vdupq_n_f32 (tol), zero = vdupq_n_f32 (0.0f);
    uint32x4_t cu = vdupq_n_u32 (0), cd = vdupq_n_u32 (0);

    for (guint p = 0; p < N_PIX; p += 4) {
      float32x4_t d = vld1q_f32 (dist + p), D = vld1q_f32 (ref + p), dr = vaddq_f32 (d, vrn);
      float32x4_t e0 = vabsq_f32 (vsubq_f32 (d, D)), e1 = vabsq_f32 (vsubq_f32 (dr, D));
      uint32x4_t valid = vcgtq_f32 (d, zero), k = vcltq_f32 (e1, e0);
      uint32x4_t dec = vandq_u32 (vandq_u32 (valid, vcgtq_f32 (D, zero)), vcleq_f32 (vminq_f32 (e0, e1), vtol));

      vst1q_f32 (dist + p, vbslq_f32 (dec, vbslq_f32 (k, dr, d), zero));
      vst1q_f32 (last + p, vbslq_f32 (valid, d, vld1q_f32 (last + p)));
      cu = vsubq_u32 (cu, vandq_u32 (dec, k));          /* a set lane is all ones = -1 */
      cd = vsubq_u32 (cd, vbicq_u32 (valid, dec));
    }
    *n_unw = vaddvq_u32 (cu);
    *n_und = vaddvq_u32 (cd);
    return TRUE;
  }
#endif
  for (guint p = 0; p < N_PIX; p++) {
    gfloat d = dist[p], D = ref[p], dr = d + rn;
    gfloat e0 = fabsf (d - D), e1 = fabsf (dr - D);
    gboolean valid = d > 0.0f, k = e1 < e0, dec = valid & (D > 0.0f) & (MIN (e0, e1) <= tol);

    dist[p] = dec ? (k ? dr : d) : 0.0f;
    last[p] = valid ? d : last[p];
    unw += dec & k;
    und += valid & !dec;
  }
  *n_unw = unw;
  *n_und = und;
  return TRUE;
}

/* Flying pixels: on a depth edge a pixel sees foreground and background, and its mixed phasor gives a depth in
 * between. A valid pixel whose depth lies strictly between both valid neighbours along a row, column or
 * diagonal, with jumps above the threshold on both sides, becomes invalid; a true edge pixel has a similar
 * neighbour on one side and stays. Decided on the unfiltered map, then cleared. */
static guint
remove_flying_pixels (GstTofDepth * self, gfloat thr)
{
  static const gint dx[4] = { 1, 0, 1, 1 }, dy[4] = { 0, 1, 1, -1 };
  const gfloat *d = self->dist;
  guint n = 0;

  memset (self->flying, 0, N_PIX);
  for (gint y = 1; y < GST_TOF_HEIGHT - 1; y++) {
    for (gint x = 1; x < GST_TOF_WIDTH - 1; x++) {
      gint p = y * GST_TOF_WIDTH + x;

      if (d[p] <= 0.0f)
        continue;
      for (guint k = 0; k < 4; k++) {
        gint o = dy[k] * GST_TOF_WIDTH + dx[k];
        gfloat a = d[p - o], b = d[p + o];

        if (a > 0.0f && b > 0.0f && ((a - d[p] > thr && d[p] - b > thr) ||
                (b - d[p] > thr && d[p] - a > thr))) {
          self->flying[p] = 1;
          n++;
          break;
        }
      }
    }
  }
  for (guint p = 0; n > 0 && p < N_PIX; p++)
    if (self->flying[p])
      self->dist[p] = 0.0f;
  return n;
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
  hi = self->max_distance > 0.0 ? (gfloat) self->max_distance :
      (gfloat) self->models[self->unwrap ? GST_TOF_RANGE_FAR : mode].r;
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
      guint8 g;

      if (fmt == GST_VIDEO_FORMAT_GRAY16_LE) {
        if (confidence)
          GST_WRITE_UINT16_LE (row + 2 * x, d[x] > 0.0f ?
              (guint16) CLAMP (lrintf (65535.0f * MIN (1.0f, a[x] / cscale)), 1, 65535) : 0);
        else
          GST_WRITE_UINT16_LE (row + 2 * x, (guint16) CLAMP (lrintf (a[x]), 0, 65535));
        continue;
      }
      if (confidence)
        g = d[x] > 0.0f ? (guint8) CLAMP (lrintf (255.0f * MIN (1.0f, a[x] / cscale)), 1, 255) : 0;
      else
        g = (guint8) CLAMP (lrintf (255.0f * a[x] / amax), 0, 255);
      switch (fmt) {
        case GST_VIDEO_FORMAT_GRAY8:
          row[x] = g;
          break;
        case GST_VIDEO_FORMAT_RGBA:
        case GST_VIDEO_FORMAT_BGRA:
          row[4 * x] = row[4 * x + 1] = row[4 * x + 2] = g;
          row[4 * x + 3] = 255;
          break;
        case GST_VIDEO_FORMAT_RGB:
        case GST_VIDEO_FORMAT_BGR:
          row[3 * x] = row[3 * x + 1] = row[3 * x + 2] = g;
          break;
        default:
          break;
      }
    }
  }
}

#ifdef TOF_HAVE_NEON
G_STATIC_ASSERT (GST_TOF_WIDTH % 8 == 0);

/* row writers shared by the NEON fills: grey value g (8 bit) in the output format */
static inline void
write_grey8_row (guint8 * row, const guint8 * g, GstVideoFormat fmt)
{
  switch (fmt) {
    case GST_VIDEO_FORMAT_GRAY8:
      memcpy (row, g, GST_TOF_WIDTH);
      break;
    case GST_VIDEO_FORMAT_RGBA:
    case GST_VIDEO_FORMAT_BGRA:
      for (guint x = 0; x < GST_TOF_WIDTH; x += 8) {
        uint8x8_t v = vld1_u8 (g + x);
        uint8x8x4_t px = { {v, v, v, vdup_n_u8 (255)} };
        vst4_u8 (row + 4 * x, px);
      }
      break;
    case GST_VIDEO_FORMAT_RGB:
    case GST_VIDEO_FORMAT_BGR:
      for (guint x = 0; x < GST_TOF_WIDTH; x += 8) {
        uint8x8_t v = vld1_u8 (g + x);
        uint8x8x3_t px = { {v, v, v} };
        vst3_u8 (row + 3 * x, px);
      }
      break;
    default:
      break;
  }
}

/* fill_depth () with the per-pixel arithmetic vectorised and the format test out of the pixel loop */
static void
fill_depth_neon (GstTofDepth * self, GstVideoFrame * vf, guint mode)
{
  GstVideoFormat fmt = GST_VIDEO_FRAME_FORMAT (vf);
  guint8 *base = GST_VIDEO_FRAME_PLANE_DATA (vf, 0);
  gint stride = GST_VIDEO_FRAME_PLANE_STRIDE (vf, 0);
  const float32x4_t zero = vdupq_n_f32 (0.0f);
  gfloat lo, hi, span;
  GstTofColormap cmap;
  gint16 idx[GST_TOF_WIDTH];
  guint8 g[GST_TOF_WIDTH];

  GST_OBJECT_LOCK (self);
  lo = (gfloat) self->min_distance;
  hi = self->max_distance > 0.0 ? (gfloat) self->max_distance :
      (gfloat) self->models[self->unwrap ? GST_TOF_RANGE_FAR : mode].r;
  cmap = self->colormap;
  GST_OBJECT_UNLOCK (self);
  span = hi > lo ? hi - lo : 1.0f;

  for (guint y = 0; y < GST_TOF_HEIGHT; y++) {
    guint8 *row = base + y * stride;
    const gfloat *d = self->dist + y * GST_TOF_WIDTH;

    if (fmt == GST_VIDEO_FORMAT_GRAY16_LE) {
      /* mm, clamped to 1..65535; 0 = invalid */
      for (guint x = 0; x < GST_TOF_WIDTH; x += 4) {
        float32x4_t dv = vld1q_f32 (d + x);
        int32x4_t mm = vmaxq_s32 (vcvtnq_s32_f32 (vmulq_f32 (dv, vdupq_n_f32 (1000.0f))), vdupq_n_s32 (1));
        uint16x4_t v = vand_u16 (vqmovun_s32 (mm), vmovn_u32 (vcgtq_f32 (dv, zero)));

        vst1_u16 ((guint16 *) (row + 2 * x), v);
      }
      continue;
    }
    /* colour index 0..255 (near = 255), -1 = invalid */
    for (guint x = 0; x < GST_TOF_WIDTH; x += 4) {
      float32x4_t dv = vld1q_f32 (d + x);
      float32x4_t t = vmulq_f32 (vdupq_n_f32 (255.0f), vsubq_f32 (vdupq_n_f32 (1.0f),
              vdivq_f32 (vsubq_f32 (dv, vdupq_n_f32 (lo)), vdupq_n_f32 (span))));
      int32x4_t v = vminq_s32 (vmaxq_s32 (vcvtnq_s32_f32 (t), vdupq_n_s32 (0)), vdupq_n_s32 (255));

      v = vbslq_s32 (vcgtq_f32 (dv, zero), v, vdupq_n_s32 (-1));
      vst1_s16 (idx + x, vmovn_s32 (v));
    }
    if (fmt == GST_VIDEO_FORMAT_GRAY8 || cmap == GST_TOF_COLORMAP_GRAY) {
      for (guint x = 0; x < GST_TOF_WIDTH; x++)
        g[x] = idx[x] < 0 ? 0 : (guint8) idx[x];
      write_grey8_row (row, g, fmt);
    } else {
      guint bpp = (fmt == GST_VIDEO_FORMAT_RGBA || fmt == GST_VIDEO_FORMAT_BGRA) ? 4 : 3;
      gboolean bgr = fmt == GST_VIDEO_FORMAT_BGRA || fmt == GST_VIDEO_FORMAT_BGR;

      for (guint x = 0; x < GST_TOF_WIDTH; x++) {
        static const guint8 black[3] = { 0, 0, 0 };
        const guint8 *c = idx[x] < 0 ? black : self->turbo[idx[x]];
        guint8 *px = row + bpp * x;

        px[0] = bgr ? c[2] : c[0];
        px[1] = c[1];
        px[2] = bgr ? c[0] : c[2];
        if (bpp == 4)
          px[3] = 255;
      }
    }
  }
}

/* fill_scalar () vectorised: amplitude (counts or scaled to 8 bit) or confidence */
static void
fill_scalar_neon (GstTofDepth * self, GstVideoFrame * vf, gboolean confidence)
{
  GstVideoFormat fmt = GST_VIDEO_FRAME_FORMAT (vf);
  guint8 *base = GST_VIDEO_FRAME_PLANE_DATA (vf, 0);
  gint stride = GST_VIDEO_FRAME_PLANE_STRIDE (vf, 0);
  const float32x4_t zero = vdupq_n_f32 (0.0f), one = vdupq_n_f32 (1.0f);
  gboolean wide = fmt == GST_VIDEO_FORMAT_GRAY16_LE;
  float32x4_t amax, cscale;
  guint8 g[GST_TOF_WIDTH];

  GST_OBJECT_LOCK (self);
  amax = vdupq_n_f32 ((gfloat) self->amplitude_max);
  cscale = vdupq_n_f32 ((gfloat) self->confidence_scale);
  GST_OBJECT_UNLOCK (self);

  for (guint y = 0; y < GST_TOF_HEIGHT; y++) {
    guint8 *row = base + y * stride;
    const gfloat *a = self->amp + y * GST_TOF_WIDTH, *d = self->dist + y * GST_TOF_WIDTH;

    for (guint x = 0; x < GST_TOF_WIDTH; x += 4) {
      float32x4_t av = vld1q_f32 (a + x);
      uint32x4_t valid = vcgtq_f32 (vld1q_f32 (d + x), zero);
      int32x4_t v;

      if (confidence) {
        float32x4_t c = vminq_f32 (one, vdivq_f32 (av, cscale));

        v = vcvtnq_s32_f32 (vmulq_f32 (vdupq_n_f32 (wide ? 65535.0f : 255.0f), c));
        v = vminq_s32 (vmaxq_s32 (v, vdupq_n_s32 (1)), vdupq_n_s32 (wide ? 65535 : 255));
        v = vandq_s32 (v, vreinterpretq_s32_u32 (valid));
      } else if (wide) {
        v = vcvtnq_s32_f32 (av);        /* counts; vqmovun clamps to 0..65535 */
      } else {
        v = vcvtnq_s32_f32 (vdivq_f32 (vmulq_f32 (vdupq_n_f32 (255.0f), av), amax));
        v = vminq_s32 (vmaxq_s32 (v, vdupq_n_s32 (0)), vdupq_n_s32 (255));
      }
      if (wide)
        vst1_u16 ((guint16 *) (row + 2 * x), vqmovun_s32 (v));
      else
        vst1_lane_u32 ((guint32 *) (g + x), vreinterpret_u32_u8 (vqmovn_u16 (vcombine_u16 (vqmovun_s32 (v),
                        vdup_n_u16 (0)))), 0);
    }
    if (!wide)
      write_grey8_row (row, g, fmt);
  }
}
#endif

/* X, Y, Z (m) per pixel, row-major, from the lens rays; NaN = invalid (organised point cloud) */
static void
fill_pointcloud (GstTofDepth * self, gfloat * out)
{
  for (guint p = 0; p < N_PIX; p++) {
    gfloat d = self->dist[p], z;

    if (d <= 0.0f) {
      out[3 * p] = out[3 * p + 1] = out[3 * p + 2] = NAN;
      continue;
    }
    z = self->dist_is_z ? d : d * self->zscale[p];
    out[3 * p] = z * self->rayx[p];
    out[3 * p + 1] = z * self->rayy[p];
    out[3 * p + 2] = z;
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
  if (out->need_caps && out == &self->outs[OUT_POINTCLOUD]) {
    /* fixed tensor caps (template + rate), a plain system-memory pool */
    GstCaps *caps = gst_caps_make_writable (gst_pad_get_pad_template_caps (out->pad));
    GstStructure *config;

    gst_caps_set_simple (caps, "rate", GST_TYPE_FRACTION, self->fps_n, self->fps_d, NULL);
    if (!gst_pad_push_event (out->pad, gst_event_new_caps (gst_caps_ref (caps)))) {
      gst_caps_unref (caps);
      GST_ELEMENT_ERROR (self, CORE, NEGOTIATION, ("Caps not accepted on %s.", GST_PAD_NAME (out->pad)),
          (NULL));
      return FALSE;
    }
    GST_INFO_OBJECT (out->pad, "caps %" GST_PTR_FORMAT, caps);
    if (out->pool) {
      gst_buffer_pool_set_active (out->pool, FALSE);
      gst_clear_object (&out->pool);
    }
    out->pool = gst_buffer_pool_new ();
    config = gst_buffer_pool_get_config (out->pool);
    gst_buffer_pool_config_set_params (config, caps, POINTCLOUD_SIZE, 2, 0);
    gst_caps_unref (caps);
    if (!gst_buffer_pool_set_config (out->pool, config) || !gst_buffer_pool_set_active (out->pool, TRUE)) {
      GST_ELEMENT_ERROR (self, RESOURCE, NO_SPACE_LEFT, ("Cannot activate the buffer pool of %s.",
              GST_PAD_NAME (out->pad)), (NULL));
      return FALSE;
    }
    out->need_caps = FALSE;
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
  if (idx == OUT_POINTCLOUD) {
    static const guint dims[GST_ML_TENSOR_MAX_DIMS] = { 1, GST_TOF_HEIGHT, GST_TOF_WIDTH, 3 };
    GstMLTensorMeta *meta;
    GstMapInfo map;

    if (!gst_buffer_map (buf, &map, GST_MAP_WRITE)) {
      gst_buffer_unref (buf);
      GST_ELEMENT_ERROR (self, RESOURCE, WRITE, ("Cannot map an output buffer."), (NULL));
      return GST_FLOW_ERROR;
    }
    fill_pointcloud (self, (gfloat *) map.data);
    gst_buffer_unmap (buf, &map);
    meta = gst_buffer_add_ml_tensor_meta (buf, GST_ML_TYPE_FLOAT32, 4, dims);
    if (meta) {
      meta->id = 0;
      meta->name = g_quark_from_static_string ("pointcloud");
    }
    gst_buffer_copy_into (buf, inbuf, GST_BUFFER_COPY_TIMESTAMPS | GST_BUFFER_COPY_FLAGS, 0, -1);
    return gst_pad_push (out->pad, buf);
  }
  if (!gst_video_frame_map (&vf, &out->info, buf, GST_MAP_WRITE)) {
    gst_buffer_unref (buf);
    GST_ELEMENT_ERROR (self, RESOURCE, WRITE, ("Cannot map an output buffer."), (NULL));
    return GST_FLOW_ERROR;
  }
#ifdef TOF_HAVE_NEON
  if (use_simd && idx == OUT_DEPTH)
    fill_depth_neon (self, &vf, mode);
  else if (use_simd)
    fill_scalar_neon (self, &vf, idx == OUT_CONFIDENCE);
  else
#endif
  if (idx == OUT_DEPTH)
    fill_depth (self, &vf, mode);
  else
    fill_scalar (self, &vf, idx == OUT_CONFIDENCE);
  gst_video_frame_unmap (&vf);
  gst_buffer_copy_into (buf, inbuf, GST_BUFFER_COPY_TIMESTAMPS | GST_BUFFER_COPY_FLAGS, 0, -1);
  return gst_pad_push (out->pad, buf);
}

/* ---------------------------------------------------------------- thread hint */

/* "0-3", "2,3", "0-1,6" (the list format of taskset -c and Cpus_allowed_list); every CPU must exist on this board */
static gboolean
parse_cpu_list (const gchar * list, cpu_set_t * set, glong * n_cpu)
{
  gchar **parts = g_strsplit (list, ",", -1);
  gboolean ok = parts[0] != NULL;

  *n_cpu = sysconf (_SC_NPROCESSORS_CONF);
  CPU_ZERO (set);
  for (guint i = 0; ok && parts[i]; i++) {
    gchar *s = g_strstrip (parts[i]), *end;
    guint64 a, b;

    if (!g_ascii_isdigit (*s)) {
      ok = FALSE;
      break;
    }
    a = b = g_ascii_strtoull (s, &end, 10);
    if (*end == '-') {
      s = end + 1;
      if (!g_ascii_isdigit (*s)) {
        ok = FALSE;
        break;
      }
      b = g_ascii_strtoull (s, &end, 10);
    }
    if (*end != '\0' || b < a || b >= (guint64) MIN (*n_cpu, CPU_SETSIZE)) {
      ok = FALSE;
      break;
    }
    for (guint64 c = a; c <= b; c++)
      CPU_SET (c, set);
  }
  g_strfreev (parts);
  return ok;
}

static gchar *
format_cpu_list (const cpu_set_t * set)
{
  GString *s = g_string_new (NULL);

  for (gint c = 0; c < CPU_SETSIZE; c++) {
    gint e = c;

    if (!CPU_ISSET (c, set))
      continue;
    while (e + 1 < CPU_SETSIZE && CPU_ISSET (e + 1, set))
      e++;
    g_string_append_printf (s, "%s%d", s->len ? "," : "", c);
    if (e > c)
      g_string_append_printf (s, "-%d", e);
    c = e;
  }
  return g_string_free (s, FALSE);
}

static gint
get_util_min (pid_t tid)
{
  TofSchedAttr a = { 0, };

  if (syscall (SYS_sched_getattr, tid, &a, sizeof (a), 0) != 0)
    return -1;
  return (gint) a.sched_util_min;
}

static gboolean
set_util_min (pid_t tid, guint32 value)
{
  TofSchedAttr a = { 0, };

  a.size = sizeof (a);
  a.sched_flags = TOF_SCHED_FLAG_KEEP_POLICY | TOF_SCHED_FLAG_KEEP_PARAMS | TOF_SCHED_FLAG_UTIL_CLAMP_MIN;
  a.sched_util_min = value;
  return syscall (SYS_sched_setattr, tid, &a, 0) == 0;
}

/* READY -> PAUSED: take the properties for this stream; an invalid CPU list stops the start */
static gboolean
thread_hint_prepare (GstTofDepth * self)
{
  gchar *list;
  glong n_cpu;
  gboolean ok = TRUE;

  GST_OBJECT_LOCK (self);
  list = g_strdup (self->cpu_affinity);
  self->hint_util_min = self->util_clamp_min;
  GST_OBJECT_UNLOCK (self);
  self->hint_cpus = list && *list;
  if (self->hint_cpus && !parse_cpu_list (list, &self->hint_set, &n_cpu)) {
    GST_ELEMENT_ERROR (self, RESOURCE, SETTINGS, ("cpu-affinity \"%s\" is not a CPU list of this board.", list),
        ("Use the format of taskset -c, e.g. \"0-3\" or \"2,3\"; this board has CPUs 0-%ld.", n_cpu - 1));
    ok = FALSE;
  }
  g_free (list);
  return ok;
}

/* undo the hint on the thread it was applied to, if that thread still exists in this process */
static void
thread_hint_restore (GstTofDepth * self)
{
  gchar *task;

  if (!self->hint_thread)
    return;
  task = g_strdup_printf ("/proc/self/task/%d", (gint) self->hint_tid);
  if (g_file_test (task, G_FILE_TEST_EXISTS)) {
    if (self->saved_cpus_valid && sched_setaffinity (self->hint_tid, sizeof (cpu_set_t), &self->saved_set) != 0)
      GST_WARNING_OBJECT (self, "thread %d: CPU set not restored: %s", (gint) self->hint_tid, g_strerror (errno));
    /* a previous 0 is the default request: reset rather than pin it */
    if (self->saved_util_min >= 0 && !set_util_min (self->hint_tid,
            self->saved_util_min ? (guint32) self->saved_util_min : TOF_UTIL_CLAMP_RESET))
      GST_WARNING_OBJECT (self, "thread %d: uclamp.min not restored: %s", (gint) self->hint_tid, g_strerror (errno));
  }
  g_free (task);
  self->hint_thread = NULL;
  self->saved_cpus_valid = FALSE;
  self->saved_util_min = -1;
  GST_OBJECT_LOCK (self);
  g_clear_pointer (&self->thread_cpus, g_free);
  self->thread_util_min = -1;
  GST_OBJECT_UNLOCK (self);
}

/* streaming thread, once per thread: apply cpu-affinity / util-clamp-min and remember what the thread had */
static void
thread_hint_apply (GstTofDepth * self)
{
  GThread *thread = g_thread_self ();
  pid_t tid;
  cpu_set_t now;
  gchar *cpus = NULL;
  gint util, err;

  if (thread == self->hint_thread)
    return;
  thread_hint_restore (self);   /* another thread took over */
  tid = gettid ();
  self->hint_thread = thread;
  self->hint_tid = tid;
  if (self->hint_cpus) {
    if (sched_getaffinity (tid, sizeof (cpu_set_t), &self->saved_set) == 0 &&
        sched_setaffinity (tid, sizeof (cpu_set_t), &self->hint_set) == 0) {
      self->saved_cpus_valid = TRUE;
    } else {
      err = errno;
      GST_ELEMENT_WARNING (self, RESOURCE, SETTINGS, ("Cannot apply cpu-affinity to the streaming thread."),
          ("sched_setaffinity: %s", g_strerror (err)));
    }
  }
  if (self->hint_util_min >= 0) {
    gint prev = get_util_min (tid);

    if (prev >= 0 && set_util_min (tid, (guint32) self->hint_util_min)) {
      self->saved_util_min = prev;
    } else {
      err = errno;
      GST_ELEMENT_WARNING (self, RESOURCE, SETTINGS, ("Cannot apply util-clamp-min to the streaming thread."),
          ("sched_setattr: %s", g_strerror (err)));
    }
  }
  if (sched_getaffinity (tid, sizeof (cpu_set_t), &now) == 0)
    cpus = format_cpu_list (&now);
  util = get_util_min (tid);
  GST_INFO_OBJECT (self, "streaming thread %d: CPUs %s, uclamp.min %d", (gint) tid, GST_STR_NULL (cpus), util);
  GST_OBJECT_LOCK (self);
  g_free (self->thread_cpus);
  self->thread_cpus = cpus;
  self->thread_util_min = util;
  GST_OBJECT_UNLOCK (self);
}

/* ---------------------------------------------------------------- pads */

/* a reference left out of the outputs (output-references=false, RUN-0110): a GAP event in its place on every linked
 * output, so sinks and aggregators downstream know that no frame comes for that time */
static GstFlowReturn
push_gaps (GstTofDepth * self, const gboolean * active, GstBuffer * inbuf)
{
  for (guint k = 0; k < N_OUT; k++) {
    if (!active[k])
      continue;
    if (!push_sticky (self, &self->outs[k]))
      return GST_FLOW_NOT_NEGOTIATED;
    if (GST_BUFFER_PTS_IS_VALID (inbuf))
      gst_pad_push_event (self->outs[k].pad, gst_event_new_gap (GST_BUFFER_PTS (inbuf),
              GST_BUFFER_DURATION (inbuf)));
  }
  return GST_FLOW_OK;
}

static GstFlowReturn
gst_tof_depth_chain (GstPad * pad, GstObject * parent, GstBuffer * inbuf)
{
  GstTofDepth *self = GST_TOF_DEPTH (parent);
  GstFlowReturn ret = GST_FLOW_OK;
  GstMapInfo map;
  gboolean active[N_OUT];
  guint mode, flying = 0, wrapped = 0, unwrapped = 0, undecided = 0;
  gfloat flying_thr, wrap_rho, uw_delta, uw_tol;
  gboolean to_z, unwrap, unwrapped_group = FALSE, drop;
  guint64 t0 = g_get_monotonic_time ();

  if (self->hint_cpus || self->hint_util_min >= 0)
    thread_hint_apply (self);
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
  flying_thr = (gfloat) self->flying_pixel_threshold;
  wrap_rho = (gfloat) self->wrap_reflectivity;
  to_z = self->depth_type == GST_TOF_DEPTH_Z && self->zscale;
  unwrap = self->unwrap;
  uw_delta = (gfloat) self->unwrap_delta;
  uw_tol = (gfloat) self->unwrap_tolerance;
  drop = !self->output_references && (GST_READ_UINT16_LE (map.data + GST_TOF_OFF_FLAGS) & GST_TOF_FLAG_REFERENCE);
  GST_OBJECT_UNLOCK (self);

  if (active[OUT_POINTCLOUD] && !self->zscale) {
    gst_buffer_unmap (inbuf, &map);
    GST_ELEMENT_ERROR (self, RESOURCE, SETTINGS, ("The pointcloud output needs a lens model."),
        ("Set lens= to a key file with a [lens] group, or add that group to the calibration file."));
    gst_buffer_unref (inbuf);
    return GST_FLOW_ERROR;
  }
  if (wrap_rho > 0.0f && (self->models[mode].wrap_k <= 0.0 || self->models[mode].wrap_sigma <= 0.0)) {
    gst_buffer_unmap (inbuf, &map);
    GST_ELEMENT_ERROR (self, RESOURCE, SETTINGS, ("wrap-reflectivity needs the calibration's wrap reference."),
        ("Add wrap_k and wrap_sigma to its [%s] group (tools/tof-wrap.py conf).",
            mode == GST_TOF_RANGE_NEAR ? "near" : "far"));
    gst_buffer_unref (inbuf);
    return GST_FLOW_ERROR;
  }
  compute (self, map.data, active[OUT_DEPTH] || active[OUT_CONFIDENCE] || active[OUT_POINTCLOUD]);
  gst_buffer_unmap (inbuf, &map);
  if (unwrap && (active[OUT_DEPTH] || active[OUT_CONFIDENCE] || active[OUT_POINTCLOUD]))
    unwrapped_group = unwrap_group (self, mode, uw_delta, uw_tol, &unwrapped, &undecided);
  if (drop) {
    ret = push_gaps (self, active, inbuf);
    goto counted;
  }
  if (wrap_rho > 0.0f && !unwrapped_group &&
      (active[OUT_DEPTH] || active[OUT_CONFIDENCE] || active[OUT_POINTCLOUD]))
    wrapped = remove_wrapped (self, &self->models[mode], wrap_rho);
  if (flying_thr > 0.0f && (active[OUT_DEPTH] || active[OUT_CONFIDENCE] || active[OUT_POINTCLOUD]))
    flying = remove_flying_pixels (self, flying_thr);
  /* the flying-pixel test judged the measured (radial) distance; 0 = invalid stays 0 */
  self->dist_is_z = to_z && active[OUT_DEPTH];
  if (self->dist_is_z)
    for (guint p = 0; p < N_PIX; p++)
      self->dist[p] *= self->zscale[p];

  for (guint k = 0; k < N_OUT; k++) {
    GstFlowReturn r;

    if (!active[k])
      continue;
    r = push_output (self, &self->outs[k], k, mode, inbuf);
    ret = gst_flow_combiner_update_pad_flow (self->combiner, self->outs[k].pad, r);
    if (ret != GST_FLOW_OK && ret != GST_FLOW_NOT_LINKED)
      break;
  }
counted:
  if (!active[OUT_DEPTH] && !active[OUT_AMPLITUDE] && !active[OUT_CONFIDENCE] && !active[OUT_POINTCLOUD])
    ret = GST_FLOW_NOT_LINKED;

  GST_OBJECT_LOCK (self);
  self->n_groups++;
  self->n_refs_dropped += drop;
  self->n_flying += flying;
  self->n_wrap += wrapped;
  self->n_unwrapped += unwrapped;
  self->n_undecided += undecided;
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
      unwrap_reset (self);
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

/* time a group spends in this element before its output is pushed: the processing-latency property,
 * or automatically from the number of linked outputs (they are computed and pushed one after the other) */
static GstClockTime
processing_latency (GstTofDepth * self)
{
  GstClockTime lat;
  guint n = 0;

  GST_OBJECT_LOCK (self);
  if (self->processing_latency >= 0) {
    lat = (GstClockTime) self->processing_latency;
  } else {
    for (guint k = 0; k < N_OUT; k++)
      n += self->outs[k].pad && gst_pad_is_linked (self->outs[k].pad);
    lat = AUTO_LATENCY_BASE + MAX (n, 1) * AUTO_LATENCY_PER_OUTPUT;
  }
  GST_OBJECT_UNLOCK (self);
  return lat;
}

static gboolean
gst_tof_depth_src_query (GstPad * pad, GstObject * parent, GstQuery * query)
{
  if (GST_QUERY_TYPE (query) == GST_QUERY_LATENCY) {
    GstTofDepth *self = GST_TOF_DEPTH (parent);
    GstClockTime min, max, lat;
    gboolean live;

    if (!gst_pad_peer_query (self->sinkpad, query))
      return FALSE;
    gst_query_parse_latency (query, &live, &min, &max);
    lat = processing_latency (self);
    GST_DEBUG_OBJECT (pad, "upstream live %d min %" GST_TIME_FORMAT " max %" GST_TIME_FORMAT
        ", + processing %" GST_TIME_FORMAT, live, GST_TIME_ARGS (min), GST_TIME_ARGS (max),
        GST_TIME_ARGS (lat));
    gst_query_set_latency (query, live, min + lat, GST_CLOCK_TIME_IS_VALID (max) ? max + lat : max);
    return TRUE;
  }
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
  guint idx = g_strcmp0 (tname, "amplitude") == 0 ? OUT_AMPLITUDE :
      g_strcmp0 (tname, "pointcloud") == 0 ? OUT_POINTCLOUD : OUT_CONFIDENCE;
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
    if (!thread_hint_prepare (self) || !load_models (self))
      return GST_STATE_CHANGE_FAILURE;
    if (self->outs[OUT_POINTCLOUD].pad && !self->zscale) {
      GST_ELEMENT_ERROR (self, RESOURCE, SETTINGS, ("The pointcloud output needs a lens model."),
          ("Set lens= to a key file with a [lens] group, or add that group to the calibration file."));
      return GST_STATE_CHANGE_FAILURE;
    }
    gst_segment_init (&self->segment, GST_FORMAT_TIME);
    gst_flow_combiner_reset (self->combiner);
    self->hist_count = self->hist_pos = 0;
    g_clear_pointer (&self->hist, g_free);
    unwrap_reset (self);
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
    /* the pads are inactive now, so no chain call runs on the hinted thread */
    thread_hint_restore (self);
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
    case PROP_FLYING_PIXEL_THRESHOLD:
      self->flying_pixel_threshold = g_value_get_double (value);
      break;
    case PROP_UNWRAP:
      self->unwrap = g_value_get_boolean (value);
      break;
    case PROP_UNWRAP_DELTA:
      self->unwrap_delta = g_value_get_double (value);
      break;
    case PROP_UNWRAP_TOLERANCE:
      self->unwrap_tolerance = g_value_get_double (value);
      break;
    case PROP_OUTPUT_REFERENCES:
      self->output_references = g_value_get_boolean (value);
      break;
    case PROP_WRAP_REFLECTIVITY:
      self->wrap_reflectivity = g_value_get_double (value);
      break;
    case PROP_DEPTH_TYPE:
      self->depth_type = g_value_get_enum (value);
      self->models_loaded = FALSE;
      break;
    case PROP_LENS:
      g_free (self->lens);
      self->lens = g_value_dup_string (value);
      self->models_loaded = FALSE;
      break;
    case PROP_PROCESSING_LATENCY:
      self->processing_latency = g_value_get_int64 (value);
      break;
    case PROP_CPU_AFFINITY:
      g_free (self->cpu_affinity);
      self->cpu_affinity = g_value_dup_string (value);
      break;
    case PROP_UTIL_CLAMP_MIN:
      self->util_clamp_min = g_value_get_int (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
  GST_OBJECT_UNLOCK (self);
  if (prop_id == PROP_PROCESSING_LATENCY)
    gst_element_post_message (GST_ELEMENT (self), gst_message_new_latency (GST_OBJECT (self)));
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
    case PROP_FLYING_PIXEL_THRESHOLD:
      g_value_set_double (value, self->flying_pixel_threshold);
      break;
    case PROP_WRAP_REFLECTIVITY:
      g_value_set_double (value, self->wrap_reflectivity);
      break;
    case PROP_DEPTH_TYPE:
      g_value_set_enum (value, self->depth_type);
      break;
    case PROP_LENS:
      g_value_set_string (value, self->lens);
      break;
    case PROP_PROCESSING_LATENCY:
      g_value_set_int64 (value, self->processing_latency);
      break;
    case PROP_CPU_AFFINITY:
      g_value_set_string (value, self->cpu_affinity);
      break;
    case PROP_UTIL_CLAMP_MIN:
      g_value_set_int (value, self->util_clamp_min);
      break;
    case PROP_UNWRAP:
      g_value_set_boolean (value, self->unwrap);
      break;
    case PROP_UNWRAP_DELTA:
      g_value_set_double (value, self->unwrap_delta);
      break;
    case PROP_UNWRAP_TOLERANCE:
      g_value_set_double (value, self->unwrap_tolerance);
      break;
    case PROP_OUTPUT_REFERENCES:
      g_value_set_boolean (value, self->output_references);
      break;
    case PROP_STATS:
      g_value_take_boxed (value, gst_structure_new ("stats",
              "groups", G_TYPE_UINT64, self->n_groups,
              "calibration", G_TYPE_STRING, self->calibration_used,
              "depth-type", G_TYPE_STRING, self->depth_type == GST_TOF_DEPTH_Z ? "z" : "radial",
              "lens", G_TYPE_STRING, self->lens_used ? self->lens_used : "",
              "flying-pixels", G_TYPE_UINT64, self->n_flying,
              "wrap-flagged", G_TYPE_UINT64, self->n_wrap,
              "unwrap-references", G_TYPE_UINT64, self->n_uw_refs,
              "unwrapped", G_TYPE_UINT64, self->n_unwrapped,
              "unwrap-undecided", G_TYPE_UINT64, self->n_undecided,
              "references-dropped", G_TYPE_UINT64, self->n_refs_dropped,
              "mean-process-us", G_TYPE_DOUBLE,
              self->n_groups ? self->proc_ns / 1000.0 / self->n_groups : 0.0,
              "thread-cpus", G_TYPE_STRING, self->thread_cpus ? self->thread_cpus : "",
              "thread-util-clamp-min", G_TYPE_INT, self->thread_util_min, NULL));
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
  g_free (self->calibration_used);
  g_free (self->lens);
  g_free (self->lens_used);
  g_free (self->cpu_affinity);
  g_free (self->thread_cpus);
  g_free (self->zscale);
  g_free (self->rayx);
  g_free (self->rayy);
  g_free (self->dist);
  g_free (self->amp);
  g_free (self->sum_i);
  g_free (self->sum_q);
  g_free (self->hist);
  g_free (self->flying);
  g_free (self->uw_ref);
  g_free (self->uw_last);
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
          "Key file with the per-mode depth model (tools/tof-sweep.py CAL_OUT=*.conf); unset = "
          "$GST_TOF_CALIBRATION or " TOF_DEFAULT_CALIBRATION " (error if missing); \"\" = uncalibrated, "
          "not metric", NULL, rw | GST_PARAM_MUTABLE_READY));
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
          "for exposure, DEC-0017); a far reference group (qtitofsrc far-reference-interval) is not averaged and "
          "leaves the window as it is", 1, MAX_AVERAGE, DEFAULT_TEMPORAL_AVERAGE,
          rw | GST_PARAM_MUTABLE_PLAYING));
  g_object_class_install_property (gobject_class, PROP_FLYING_PIXEL_THRESHOLD,
      g_param_spec_double ("flying-pixel-threshold", "Flying-pixel threshold",
          "Invalidate edge pixels whose depth lies between both neighbours along a row, column or diagonal "
          "with a jump above this (m) on each side (mixed foreground/background); 0 = off", 0.0, 1.0,
          DEFAULT_FLYING_PIXEL_THRESHOLD, rw | GST_PARAM_MUTABLE_PLAYING));
  g_object_class_install_property (gobject_class, PROP_WRAP_REFLECTIVITY,
      g_param_spec_double ("wrap-reflectivity", "Wrap reflectivity",
          "Invalidate pixels that may lie one range further (beyond 2 m near / 4 m far): those whose amplitude "
          "a target this bright, relative to the calibration's wrap reference (wrap_k, wrap_sigma of each mode), "
          "could give from there; 2 = the measured setting (RUN-0090), 0 = off", 0.0, 100.0,
          DEFAULT_WRAP_REFLECTIVITY, rw | GST_PARAM_MUTABLE_PLAYING));
  g_object_class_install_property (gobject_class, PROP_UNWRAP,
      g_param_spec_boolean ("unwrap", "Unwrap near mode",
          "Use every far group as a reference (qtitofsrc far-reference-interval) and output near-mode pixels beyond "
          "the near range at r + R_near; pixels whose reading no longer fits their reference become invalid until "
          "the next one (RUN-0105). Far groups are output as they are (see output-references)", FALSE, rw | GST_PARAM_MUTABLE_PLAYING));
  g_object_class_install_property (gobject_class, PROP_UNWRAP_DELTA,
      g_param_spec_double ("unwrap-delta", "Unwrap reference limit",
          "Largest |r_near + k R_near - r_far| (m) for which a far group sets a pixel's reference", 0.01, 2.0,
          DEFAULT_UNWRAP_DELTA, rw | GST_PARAM_MUTABLE_PLAYING));
  g_object_class_install_property (gobject_class, PROP_UNWRAP_TOLERANCE,
      g_param_spec_double ("unwrap-tolerance", "Unwrap tolerance",
          "Largest |r + k R_near - reference| (m) for which a near pixel is output; above it the pixel moved and is "
          "invalid until the next reference", 0.01, 1.0, DEFAULT_UNWRAP_TOLERANCE, rw | GST_PARAM_MUTABLE_PLAYING));
  g_object_class_install_property (gobject_class, PROP_OUTPUT_REFERENCES,
      g_param_spec_boolean ("output-references", "Output references",
          "Push the far reference groups (flagged by qtitofsrc far-reference-interval) to the outputs; false = unwrap "
          "still uses them, but they are left out with a GAP event in their place, so the outputs carry the near "
          "groups only", DEFAULT_OUTPUT_REFERENCES, rw | GST_PARAM_MUTABLE_PLAYING));
  g_object_class_install_property (gobject_class, PROP_DEPTH_TYPE,
      g_param_spec_enum ("depth-type", "Depth type", "Distance definition of the depth output: radial "
          "(along each pixel's ray, as calibrated) or z (along the optical axis, needs a lens model)",
          gst_tof_depth_type_get_type (), GST_TOF_DEPTH_RADIAL, rw | GST_PARAM_MUTABLE_READY));
  g_object_class_install_property (gobject_class, PROP_LENS,
      g_param_spec_string ("lens", "Lens model",
          "Key file with a [lens] group (f, cx, cy, k1, optional k2: pinhole + inverse radial distortion, "
          "tools/tof-lens.py); unset = the calibration file's [lens] group, if it has one. Needed by depth-type=z",
          NULL, rw | GST_PARAM_MUTABLE_READY));
  g_object_class_install_property (gobject_class, PROP_PROCESSING_LATENCY,
      g_param_spec_int64 ("processing-latency", "Processing latency",
          "Time (ns) from a group's arrival to its outputs, added to the latency query; -1 = automatic "
          "(4 ms + 2.5 ms per linked output)", -1, GST_SECOND, DEFAULT_PROCESSING_LATENCY,
          rw | GST_PARAM_MUTABLE_PLAYING));
  g_object_class_install_property (gobject_class, PROP_CPU_AFFINITY,
      g_param_spec_string ("cpu-affinity", "CPU affinity",
          "CPUs the streaming thread that runs this element may use, a list as for taskset -c (\"0-3\" = the "
          "Cortex-A78C cores of the VENTUNO Q); without a queue in front this is also qtitofsrc's thread. Applied when "
          "the first group arrives, undone at stop; unset = the thread is left alone", NULL,
          rw | GST_PARAM_MUTABLE_READY));
  g_object_class_install_property (gobject_class, PROP_UTIL_CLAMP_MIN,
      g_param_spec_int ("util-clamp-min", "Utilisation clamp minimum",
          "Minimum utilisation (0-1024) the scheduler assumes for the streaming thread that runs this element "
          "(Linux uclamp.min): it prefers cores of at least this capacity and schedutil keeps their clock up to it. "
          "Applied when the first group arrives, undone at stop; -1 = the thread is left alone", -1, 1024,
          DEFAULT_UTIL_CLAMP_MIN, rw | GST_PARAM_MUTABLE_READY));
  g_object_class_install_property (gobject_class, PROP_STATS,
      g_param_spec_boxed ("stats", "Statistics", "groups, calibration (file in use, \"\" = none), depth-type, lens (file in use, \"\" = none), "
          "flying-pixels (removed in all), wrap-flagged (removed in all), unwrap-references (far groups used), unwrapped "
          "(pixels output beyond R_near), unwrap-undecided (valid near pixels removed), references-dropped (references "
          "left out, output-references=false), mean-process-us, thread-cpus and "
          "thread-util-clamp-min (the streaming thread's, read back after applying a hint; \"\" / -1 = no hint)",
          GST_TYPE_STRUCTURE, G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

  gst_element_class_add_static_pad_template (element_class, &sink_template);
  gst_element_class_add_static_pad_template (element_class, &depth_template);
  gst_element_class_add_static_pad_template (element_class, &amplitude_template);
  gst_element_class_add_static_pad_template (element_class, &confidence_template);
  {
    /* the QIM SDK tensor format, built as its ML elements do (gst_ml_info_to_caps) + rate */
    GstMLInfo info;
    GstCaps *caps;

    gst_ml_info_init (&info);
    info.type = GST_ML_TYPE_FLOAT32;
    info.n_tensors = 1;
    info.n_dimensions[0] = 4;
    info.tensors[0][0] = 1;
    info.tensors[0][1] = GST_TOF_HEIGHT;
    info.tensors[0][2] = GST_TOF_WIDTH;
    info.tensors[0][3] = 3;
    caps = gst_ml_info_to_caps (&info);
    gst_caps_set_simple (caps, "rate", GST_TYPE_FRACTION_RANGE, 0, 1, 1000, 1, NULL);
    gst_element_class_add_pad_template (element_class,
        gst_pad_template_new ("pointcloud", GST_PAD_SRC, GST_PAD_REQUEST, caps));
    gst_caps_unref (caps);
  }
  gst_element_class_set_static_metadata (element_class, "Arducam ToF depth",
      "Filter/Converter/Video", "Calibrated depth, amplitude and confidence from groups of "
      "4 DCS frames (qtitofsrc)", "arducam-tof-ventuno project");

  element_class->request_new_pad = GST_DEBUG_FUNCPTR (gst_tof_depth_request_new_pad);
  element_class->release_pad = GST_DEBUG_FUNCPTR (gst_tof_depth_release_pad);
  element_class->change_state = GST_DEBUG_FUNCPTR (gst_tof_depth_change_state);

  GST_DEBUG_CATEGORY_INIT (gst_tof_depth_debug, "qtitofdepth", 0, "Arducam ToF depth");
#ifdef TOF_HAVE_NEON
  use_simd = g_getenv ("GST_TOF_DEPTH_SCALAR") == NULL;
  GST_INFO ("per-pixel path: %s", use_simd ? "NEON" : "scalar (GST_TOF_DEPTH_SCALAR)");
#endif
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
  self->flying_pixel_threshold = DEFAULT_FLYING_PIXEL_THRESHOLD;
  self->wrap_reflectivity = DEFAULT_WRAP_REFLECTIVITY;
  self->unwrap_delta = DEFAULT_UNWRAP_DELTA;
  self->unwrap_tolerance = DEFAULT_UNWRAP_TOLERANCE;
  self->output_references = DEFAULT_OUTPUT_REFERENCES;
  self->depth_type = GST_TOF_DEPTH_RADIAL;
  self->processing_latency = DEFAULT_PROCESSING_LATENCY;
  self->util_clamp_min = DEFAULT_UTIL_CLAMP_MIN;
  self->hint_util_min = self->saved_util_min = self->thread_util_min = -1;
  self->fps_n = 30;
  self->fps_d = 1;
  self->hist_mode = -1;
  self->dist = g_new0 (gfloat, N_PIX);
  self->amp = g_new0 (gfloat, N_PIX);
  self->sum_i = g_new0 (gfloat, N_PIX);
  self->sum_q = g_new0 (gfloat, N_PIX);
  self->flying = g_new0 (guint8, N_PIX);
  self->uw_ref = g_new0 (gfloat, N_PIX);
  self->uw_last = g_new0 (gfloat, N_PIX);
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
