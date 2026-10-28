/*
 * Copyright (c) 2026, arducam-tof-ventuno project.
 * SPDX-License-Identifier: BSD-3-Clause-Clear
 *
 * qtitofsrc: Arducam ToF camera (arducam-pivariety sensor driver) on the CAMSS RDI path.
 *
 * Finds the qcom-camss media device, links csiphyN -> csidN -> vfeN_rdi0 and sets 240x180
 * SRGGB12_1X12 on every pad, holds the CSI mux line if the board has one, streams the RDI node (MMAP)
 * and assembles the module's frames into groups of 4 DCS frames: the module sends 4 frames in a burst
 * at 30 groups/s (120 frames/s), so a group starts after a gap of more than 10 ms and is 4 frames with
 * consecutive sequence numbers. Each group is pushed as one DCS16 buffer (tofdcs.h). The data has no
 * scene-independent mode signature, so the header carries the requested mode and groups that may still
 * be in another mode are dropped: frames before the first gap after STREAMON and the next
 * STREAMON_DROP_GROUPS groups (the module starts in its default mode), and after a range write every
 * group whose first frame starts less than settle-time after the write (measured: groups starting
 * <= 17 ms after it are in the old mode, 37-39 ms old, mixed or blank, >= 49 ms in the new mode).
 *
 * far-reference-interval: with range=near, after N pushed near groups the source writes far and, after
 * the next pushed far group, near again (the same write and settle-time drop as a range change). That
 * far group carries GST_TOF_FLAG_REFERENCE: qtitofdepth unwrap decides from it per pixel whether a near
 * reading is r or r + R. The range property keeps the requested mode; its read-back from the driver
 * shows far during a reference.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "tofsrc.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <linux/gpio.h>
#include <linux/media.h>
#include <linux/media-bus-format.h>
#include <linux/v4l2-subdev.h>
#include <linux/videodev2.h>

GST_DEBUG_CATEGORY_STATIC (gst_tof_src_debug);
#define GST_CAT_DEFAULT gst_tof_src_debug

/* Sensor controls of arducam-pivariety: trigger_mode (range) and frame_rate; the read-only
 * firmware_version and device_id are optional (start works without them, reported as 0) */
#define CID_TRIGGER_MODE      0x00981901
#define CID_FRAME_RATE        0x00981906
#define CID_FIRMWARE_VERSION  0x00981a00
#define CID_DEVICE_ID         0x00981a01

#define RAW_LINE_BYTES        360
#define GROUP_GAP_NS          (10 * GST_MSECOND)
#define GROUP_DURATION        (GST_SECOND / 30)
/* a group is complete 11.6 ms after its first frame (4-phase burst) and pushed <= 14.9 ms after it
 * (measured on a QCS8300): the min latency, not a whole group period */
#define GROUP_LATENCY         (15 * GST_MSECOND)
#define NO_FRAME_TIMEOUT      GST_SECOND
#define FIRMWARE_IGNORES_EXPOSURE 0x10002

#define DEFAULT_CSIPHY        0
#define DEFAULT_SETUP         TRUE
#define DEFAULT_MUX_GPIO      "auto"
#define DEFAULT_RANGE         GST_TOF_RANGE_FAR
#define DEFAULT_EXPOSURE      800
#define DEFAULT_FRAME_RATE    60
#define DEFAULT_V4L2_BUFFERS  8
#define DEFAULT_SETTLE_MS     60
#define STREAMON_DROP_GROUPS  2

enum
{
  PROP_0,
  PROP_MEDIA_DEVICE,
  PROP_DEVICE,
  PROP_CSIPHY,
  PROP_SETUP_PIPELINE,
  PROP_MUX_GPIO,
  PROP_RANGE,
  PROP_EXPOSURE,
  PROP_FRAME_RATE,
  PROP_FIRMWARE_VERSION,
  PROP_DEVICE_ID,
  PROP_NUM_V4L2_BUFFERS,
  PROP_SETTLE_TIME,
  PROP_FAR_REFERENCE_INTERVAL,
  PROP_STATS,
};

typedef struct
{
  guint8 *start;
  gsize length;
} TofMmap;

struct _GstTofSrc
{
  GstPushSrc parent;

  /* properties */
  gchar *media_device;
  gchar *device;
  guint csiphy;
  gboolean setup_pipeline;
  gchar *mux_gpio;
  GstTofRange range;
  gint exposure;
  gint frame_rate;
  gboolean exposure_set;
  gboolean frame_rate_set;
  guint n_v4l2_buffers;
  guint settle_ms;
  guint far_ref_interval;       /* near groups between far references, 0 = off */

  /* device state (object lock for subdev_fd users outside the streaming thread) */
  gint video_fd;
  gint subdev_fd;
  gint gpio_fd;
  gchar *video_path;
  gchar *subdev_path;
  TofMmap *bufs;
  guint n_bufs;
  guint stride;
  guint32 firmware_version;
  guint32 device_id;
  gboolean warned_ignored;
  gboolean streaming;
  GstPoll *poll;                /* element lifetime, so unlock() never races stop() */
  GstPollFD pollfd;

  /* group assembly (streaming thread) */
  GstBuffer *pending;
  GstMapInfo pmap;
  guint count;
  guint32 seqs[GST_TOF_PHASES];
  guint64 ts[GST_TOF_PHASES];
  guint64 last_ts;
  guint32 last_seq;
  gboolean have_last;
  guint32 group_seq;
  guint64 last_frame_wall;

  /* mode bookkeeping and stats (object lock) */
  GstTofRange applied_range;
  guint drop_left;              /* groups still to drop after STREAMON */
  guint64 drop_before_ts;       /* drop groups whose F0 starts before this (monotonic ns) */
  gboolean after_switch;
  guint64 n_pushed;
  guint64 n_incomplete;
  guint64 n_stray;
  guint64 n_seq_gaps;
  guint64 n_switch_dropped;
  guint64 n_switches;
  guint ref_near_count;         /* near groups pushed since the last far reference */
  gboolean in_reference;        /* far written for a reference, not pushed yet */
  guint64 n_references;
};

G_DEFINE_TYPE (GstTofSrc, gst_tof_src, GST_TYPE_PUSH_SRC);

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS (GST_TOF_DCS_CAPS));

static int
xioctl (int fd, unsigned long req, void *arg)
{
  int r;

  do
    r = ioctl (fd, req, arg);
  while (r < 0 && errno == EINTR);
  return r;
}

static guint64
monotonic_ns (void)
{
  struct timespec t;

  clock_gettime (CLOCK_MONOTONIC, &t);
  return GST_TIMESPEC_TO_TIME (t);
}

/* /dev path of a character device from /sys/dev/char/MAJ:MIN/uevent (DEVNAME=...) */
static gchar *
devnode_path (guint32 major, guint32 minor)
{
  gchar *uevent = g_strdup_printf ("/sys/dev/char/%u:%u/uevent", major, minor);
  gchar *contents = NULL, *path = NULL;

  if (g_file_get_contents (uevent, &contents, NULL, NULL)) {
    gchar **lines = g_strsplit (contents, "\n", -1);

    for (gchar ** l = lines; *l; l++)
      if (g_str_has_prefix (*l, "DEVNAME="))
        path = g_strdup_printf ("/dev/%s", *l + 8);
    g_strfreev (lines);
  }
  g_free (contents);
  g_free (uevent);
  return path;
}

static gint
open_media_device (GstTofSrc * self)
{
  struct media_device_info info;
  gint fd;

  if (self->media_device) {
    fd = open (self->media_device, O_RDWR | O_CLOEXEC);
    if (fd < 0)
      GST_ERROR_OBJECT (self, "cannot open %s: %s", self->media_device,
          g_strerror (errno));
    return fd;
  }
  for (guint i = 0; i < 64; i++) {
    gchar *path = g_strdup_printf ("/dev/media%u", i);

    fd = open (path, O_RDWR | O_CLOEXEC);
    if (fd >= 0) {
      memset (&info, 0, sizeof (info));
      if (xioctl (fd, MEDIA_IOC_DEVICE_INFO, &info) == 0 &&
          g_strcmp0 (info.driver, "qcom-camss") == 0) {
        GST_INFO_OBJECT (self, "media device %s (%s)", path, info.model);
        g_free (path);
        return fd;
      }
      close (fd);
    }
    g_free (path);
  }
  return -1;
}

/* entity whose name equals @name, or starts with @name when @prefix */
static gboolean
find_entity (gint media_fd, const gchar * name, gboolean prefix,
    struct media_entity_desc *out)
{
  struct media_entity_desc e;

  memset (&e, 0, sizeof (e));
  e.id = MEDIA_ENT_ID_FLAG_NEXT;
  while (xioctl (media_fd, MEDIA_IOC_ENUM_ENTITIES, &e) == 0) {
    if (prefix ? g_str_has_prefix (e.name, name) : g_strcmp0 (e.name, name) == 0) {
      *out = e;
      return TRUE;
    }
    e.id |= MEDIA_ENT_ID_FLAG_NEXT;
  }
  return FALSE;
}

static gboolean
enable_link (GstTofSrc * self, gint media_fd, const struct media_entity_desc *src,
    guint16 src_pad, const struct media_entity_desc *sink, guint16 sink_pad)
{
  struct media_link_desc l;

  memset (&l, 0, sizeof (l));
  l.source.entity = src->id;
  l.source.index = src_pad;
  l.source.flags = MEDIA_PAD_FL_SOURCE;
  l.sink.entity = sink->id;
  l.sink.index = sink_pad;
  l.sink.flags = MEDIA_PAD_FL_SINK;
  l.flags = MEDIA_LNK_FL_ENABLED;
  if (xioctl (media_fd, MEDIA_IOC_SETUP_LINK, &l) < 0) {
    GST_ERROR_OBJECT (self, "link %s:%u -> %s:%u: %s", src->name, src_pad,
        sink->name, sink_pad, g_strerror (errno));
    return FALSE;
  }
  return TRUE;
}

static gboolean
set_pad_format (GstTofSrc * self, const struct media_entity_desc *e, guint pad)
{
  struct v4l2_subdev_format f;
  gchar *path = devnode_path (e->dev.major, e->dev.minor);
  gint fd;
  gboolean ok = FALSE;

  if (!path) {
    GST_ERROR_OBJECT (self, "no device node for %s", e->name);
    return FALSE;
  }
  fd = open (path, O_RDWR | O_CLOEXEC);
  if (fd >= 0) {
    memset (&f, 0, sizeof (f));
    f.which = V4L2_SUBDEV_FORMAT_ACTIVE;
    f.pad = pad;
    f.format.width = GST_TOF_WIDTH;
    f.format.height = GST_TOF_HEIGHT;
    f.format.code = MEDIA_BUS_FMT_SRGGB12_1X12;
    f.format.field = V4L2_FIELD_NONE;
    ok = xioctl (fd, VIDIOC_SUBDEV_S_FMT, &f) == 0 &&
        f.format.width == GST_TOF_WIDTH && f.format.height == GST_TOF_HEIGHT &&
        f.format.code == MEDIA_BUS_FMT_SRGGB12_1X12;
    if (!ok)
      GST_ERROR_OBJECT (self, "format on %s pad %u: %s (got %ux%u 0x%x)",
          e->name, pad, g_strerror (errno), f.format.width, f.format.height,
          f.format.code);
    close (fd);
  } else {
    GST_ERROR_OBJECT (self, "cannot open %s: %s", path, g_strerror (errno));
  }
  g_free (path);
  return ok;
}

/* Boards with a CSI mux in front of a connector: mux-gpio=auto holds the line that routes it to csiphy */
static const struct
{
  const gchar *compatible;      /* any entry of the device tree root's compatible */
  guint csiphy;
  const gchar *chip_label;      /* gpiochip label, the index depends on probe order */
  guint line;
  guint value;
} known_muxes[] = {
  /* Arduino VENTUNO Q: TLMM GPIO 24 high routes the CAMERA0 connector to CSIPHY0; at 0 no frame reaches CSID */
  { "arduino,monza", 0, "f100000.pinctrl", 24, 1 },
};

static gboolean
board_is_compatible (const gchar * compatible)
{
  gchar *contents = NULL;
  gsize len = 0;
  gboolean found = FALSE;

  if (!g_file_get_contents ("/sys/firmware/devicetree/base/compatible", &contents, &len, NULL))
    return FALSE;
  /* NUL-separated list */
  for (gsize i = 0; i < len && !found; i += strlen (contents + i) + 1)
    found = g_strcmp0 (contents + i, compatible) == 0;
  g_free (contents);
  return found;
}

/* "gpiochipN" of the chip labelled label, NULL if none */
static gchar *
find_gpiochip (const gchar * label)
{
  GDir *dir;
  const gchar *entry;
  gchar *found = NULL;

  if (!(dir = g_dir_open ("/dev", 0, NULL)))
    return NULL;
  while (!found && (entry = g_dir_read_name (dir))) {
    struct gpiochip_info info;
    gchar *path;
    gint fd;

    if (!g_str_has_prefix (entry, "gpiochip"))
      continue;
    path = g_build_filename ("/dev", entry, NULL);
    fd = open (path, O_RDONLY | O_CLOEXEC);
    g_free (path);
    if (fd < 0)
      continue;
    memset (&info, 0, sizeof (info));
    if (xioctl (fd, GPIO_GET_CHIPINFO_IOCTL, &info) == 0 && g_strcmp0 (info.label, label) == 0)
      found = g_strdup (entry);
    close (fd);
  }
  g_dir_close (dir);
  return found;
}

/* mux-gpio=auto: the known_muxes line for this board and csiphy as "gpiochipN:LINE=VALUE", NULL if none */
static gchar *
auto_mux_gpio (GstTofSrc * self)
{
  for (guint i = 0; i < G_N_ELEMENTS (known_muxes); i++) {
    gchar *chip, *spec;

    if (known_muxes[i].csiphy != self->csiphy || !board_is_compatible (known_muxes[i].compatible))
      continue;
    if (!(chip = find_gpiochip (known_muxes[i].chip_label))) {
      GST_WARNING_OBJECT (self, "mux-gpio auto: %s, but no gpiochip labelled %s",
          known_muxes[i].compatible, known_muxes[i].chip_label);
      return NULL;
    }
    spec = g_strdup_printf ("%s:%u=%u", chip, known_muxes[i].line, known_muxes[i].value);
    GST_INFO_OBJECT (self, "mux-gpio auto: %s csiphy %u -> %s (%s line %u = %u)",
        known_muxes[i].compatible, self->csiphy, spec, known_muxes[i].chip_label,
        known_muxes[i].line, known_muxes[i].value);
    g_free (chip);
    return spec;
  }
  GST_INFO_OBJECT (self, "mux-gpio auto: no known CSI mux for this board and csiphy %u", self->csiphy);
  return NULL;
}

/* mux-gpio "gpiochipN:LINE=VALUE", "auto" or ""; returns the line request fd, -1 if not held */
static gint
hold_gpio (GstTofSrc * self, const gchar * spec)
{
  struct gpio_v2_line_request req;
  gchar chip[32];
  guint line, value;
  gchar *path;
  gint fd;

  if (!spec || !*spec)
    return -1;
  if (g_strcmp0 (spec, "auto") == 0) {
    gchar *resolved = auto_mux_gpio (self);

    fd = resolved ? hold_gpio (self, resolved) : -1;
    g_free (resolved);
    return fd;
  }
  if (sscanf (spec, "%31[^:]:%u=%u", chip, &line, &value) != 3) {
    GST_WARNING_OBJECT (self, "mux-gpio '%s' not understood (gpiochipN:LINE=VALUE)", spec);
    return -1;
  }
  path = g_strdup_printf ("/dev/%s", chip);
  fd = open (path, O_RDWR | O_CLOEXEC);
  if (fd < 0) {
    GST_WARNING_OBJECT (self, "cannot open %s: %s", path, g_strerror (errno));
    g_free (path);
    return -1;
  }
  memset (&req, 0, sizeof (req));
  req.offsets[0] = line;
  req.num_lines = 1;
  g_strlcpy (req.consumer, "qtitofsrc", sizeof (req.consumer));
  req.config.flags = GPIO_V2_LINE_FLAG_OUTPUT;
  req.config.num_attrs = 1;
  req.config.attrs[0].attr.id = GPIO_V2_LINE_ATTR_ID_OUTPUT_VALUES;
  req.config.attrs[0].attr.values = value ? 1 : 0;
  req.config.attrs[0].mask = 1;
  if (xioctl (fd, GPIO_V2_GET_LINE_IOCTL, &req) < 0) {
    /* another process may already hold it at the right level */
    GST_WARNING_OBJECT (self, "cannot hold %s line %u: %s - continuing", path, line,
        g_strerror (errno));
    close (fd);
    g_free (path);
    return -1;
  }
  close (fd);
  GST_INFO_OBJECT (self, "holding %s line %u = %u", path, line, value);
  g_free (path);
  return req.fd;
}

static gboolean
read_ctrl (gint fd, guint32 id, gint32 * value)
{
  struct v4l2_control c = {.id = id };

  if (fd < 0 || xioctl (fd, VIDIOC_G_CTRL, &c) < 0)
    return FALSE;
  *value = c.value;
  return TRUE;
}

static gboolean
write_ctrl (GstTofSrc * self, guint32 id, gint32 value, const gchar * name)
{
  struct v4l2_control c = {.id = id,.value = value };

  if (self->subdev_fd < 0)
    return FALSE;
  if (xioctl (self->subdev_fd, VIDIOC_S_CTRL, &c) < 0) {
    GST_WARNING_OBJECT (self, "setting %s = %d failed: %s", name, value,
        g_strerror (errno));
    return FALSE;
  }
  GST_DEBUG_OBJECT (self, "%s = %d", name, value);
  return TRUE;
}

/* call with the object lock held */
static void
warn_ignored (GstTofSrc * self, const gchar * what)
{
  if (self->firmware_version == FIRMWARE_IGNORES_EXPOSURE && !self->warned_ignored) {
    GST_WARNING_OBJECT (self, "%s is accepted but ignored by module firmware "
        "0x%x: the module runs at a fixed 120 frames/s and exposure",
        what, self->firmware_version);
    self->warned_ignored = TRUE;
  }
}

/* call with the object lock held */
static void
apply_mode (GstTofSrc * self, GstTofRange mode)
{
  guint64 t = monotonic_ns ();

  if (write_ctrl (self, CID_TRIGGER_MODE, mode, "trigger_mode")) {
    self->drop_before_ts = t + self->settle_ms * GST_MSECOND;
    self->after_switch = TRUE;
    self->n_switches++;
    self->applied_range = mode;
  }
}

/* call with the object lock held */
static void
apply_range (GstTofSrc * self)
{
  self->in_reference = FALSE;
  self->ref_near_count = 0;
  apply_mode (self, self->range);
}

static void
release_device (GstTofSrc * self)
{
  if (self->video_fd >= 0 && self->streaming) {
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;

    xioctl (self->video_fd, VIDIOC_STREAMOFF, &type);
  }
  for (guint i = 0; i < self->n_bufs; i++)
    if (self->bufs[i].start && self->bufs[i].start != MAP_FAILED)
      munmap (self->bufs[i].start, self->bufs[i].length);
  g_clear_pointer (&self->bufs, g_free);
  self->n_bufs = 0;
  if (self->video_fd >= 0) {
    struct v4l2_requestbuffers req = {.count = 0,
      .type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE,.memory = V4L2_MEMORY_MMAP
    };
    xioctl (self->video_fd, VIDIOC_REQBUFS, &req);
    close (self->video_fd);
    self->video_fd = -1;
  }
  GST_OBJECT_LOCK (self);
  if (self->subdev_fd >= 0) {
    close (self->subdev_fd);
    self->subdev_fd = -1;
  }
  GST_OBJECT_UNLOCK (self);
  if (self->gpio_fd >= 0) {
    close (self->gpio_fd);
    self->gpio_fd = -1;
  }
  if (self->pending) {
    gst_buffer_unmap (self->pending, &self->pmap);
    gst_clear_buffer (&self->pending);
  }
  if (self->pollfd.fd >= 0) {
    gst_poll_remove_fd (self->poll, &self->pollfd);
    gst_poll_fd_init (&self->pollfd);
  }
  self->streaming = FALSE;
  g_clear_pointer (&self->video_path, g_free);
  g_clear_pointer (&self->subdev_path, g_free);
}

static gboolean
gst_tof_src_start (GstBaseSrc * basesrc)
{
  GstTofSrc *self = GST_TOF_SRC (basesrc);
  struct media_entity_desc sensor, csiphy, csid, rdi, video;
  struct v4l2_capability cap;
  struct v4l2_format fmt;
  struct v4l2_requestbuffers req;
  gchar name[32];
  gint media_fd;
  gint32 v;

  media_fd = open_media_device (self);
  if (media_fd < 0) {
    GST_ELEMENT_ERROR (self, RESOURCE, NOT_FOUND, ("No qcom-camss media device."),
        ("Load qcom-camss and arducam-pivariety (ToF boot entry)."));
    return FALSE;
  }
  if (!find_entity (media_fd, "arducam-pivariety", TRUE, &sensor)) {
    close (media_fd);
    GST_ELEMENT_ERROR (self, RESOURCE, NOT_FOUND, ("ToF sensor not in the media graph."),
        ("No arducam-pivariety entity; is the sensor driver bound?"));
    return FALSE;
  }
  g_snprintf (name, sizeof (name), "msm_csiphy%u", self->csiphy);
  if (!find_entity (media_fd, name, FALSE, &csiphy))
    goto no_entity;
  g_snprintf (name, sizeof (name), "msm_csid%u", self->csiphy);
  if (!find_entity (media_fd, name, FALSE, &csid))
    goto no_entity;
  g_snprintf (name, sizeof (name), "msm_vfe%u_rdi0", self->csiphy);
  if (!find_entity (media_fd, name, FALSE, &rdi))
    goto no_entity;
  g_snprintf (name, sizeof (name), "msm_vfe%u_video0", self->csiphy);
  if (!find_entity (media_fd, name, FALSE, &video))
    goto no_entity;

  if (self->setup_pipeline) {
    if (!enable_link (self, media_fd, &csiphy, 1, &csid, 0) ||
        !enable_link (self, media_fd, &csid, 1, &rdi, 0) ||
        !set_pad_format (self, &sensor, 0) || !set_pad_format (self, &csiphy, 0) ||
        !set_pad_format (self, &csid, 0) || !set_pad_format (self, &rdi, 0)) {
      close (media_fd);
      GST_ELEMENT_ERROR (self, RESOURCE, SETTINGS, ("Cannot configure the CAMSS path."),
          ("See the debug log (GST_DEBUG=qtitofsrc:5)."));
      return FALSE;
    }
  }
  close (media_fd);

  /* controls on the sensor subdev; the driver applies them at stream on (it powers the module only while
   * streaming) */
  self->subdev_path = devnode_path (sensor.dev.major, sensor.dev.minor);
  GST_OBJECT_LOCK (self);
  self->subdev_fd = self->subdev_path ? open (self->subdev_path, O_RDWR | O_CLOEXEC) : -1;
  if (self->subdev_fd < 0) {
    GST_OBJECT_UNLOCK (self);
    GST_ELEMENT_ERROR (self, RESOURCE, OPEN_READ_WRITE, ("Cannot open the sensor subdev."),
        ("%s: %s", GST_STR_NULL (self->subdev_path), g_strerror (errno)));
    return FALSE;
  }
  self->firmware_version = read_ctrl (self->subdev_fd, CID_FIRMWARE_VERSION, &v) ? (guint32) v : 0;
  self->device_id = read_ctrl (self->subdev_fd, CID_DEVICE_ID, &v) ? (guint32) v : 0;
  if (!self->firmware_version)
    GST_WARNING_OBJECT (self, "driver has no firmware_version control: firmware-version and device-id stay 0");
  apply_range (self);
  self->drop_before_ts = 0;     /* covered by the drop after STREAMON */
  self->drop_left = STREAMON_DROP_GROUPS;
  self->after_switch = FALSE;
  self->n_switches = 0;
  self->n_references = 0;
  if (self->exposure_set) {
    write_ctrl (self, V4L2_CID_EXPOSURE, self->exposure, "exposure");
    warn_ignored (self, "exposure");
  }
  if (self->frame_rate_set) {
    write_ctrl (self, CID_FRAME_RATE, self->frame_rate, "frame_rate");
    warn_ignored (self, "frame-rate");
  }
  GST_OBJECT_UNLOCK (self);
  GST_INFO_OBJECT (self, "sensor %s (%s): device id 0x%04x, firmware 0x%x",
      sensor.name, self->subdev_path, self->device_id, self->firmware_version);

  self->gpio_fd = hold_gpio (self, self->mux_gpio);

  /* RDI video node */
  self->video_path = self->device ? g_strdup (self->device) :
      devnode_path (video.dev.major, video.dev.minor);
  self->video_fd = self->video_path ?
      open (self->video_path, O_RDWR | O_NONBLOCK | O_CLOEXEC) : -1;
  if (self->video_fd < 0) {
    GST_ELEMENT_ERROR (self, RESOURCE, OPEN_READ, ("Cannot open the RDI video node."),
        ("%s: %s", GST_STR_NULL (self->video_path), g_strerror (errno)));
    goto fail;
  }
  memset (&cap, 0, sizeof (cap));
  if (xioctl (self->video_fd, VIDIOC_QUERYCAP, &cap) < 0 ||
      !(cap.device_caps & V4L2_CAP_VIDEO_CAPTURE_MPLANE)) {
    GST_ELEMENT_ERROR (self, RESOURCE, SETTINGS, ("%s is not a multi-planar capture node.",
            self->video_path), (NULL));
    goto fail;
  }
  memset (&fmt, 0, sizeof (fmt));
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  fmt.fmt.pix_mp.width = GST_TOF_WIDTH;
  fmt.fmt.pix_mp.height = GST_TOF_HEIGHT;
  fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_SRGGB12P;
  fmt.fmt.pix_mp.field = V4L2_FIELD_NONE;
  fmt.fmt.pix_mp.num_planes = 1;
  if (xioctl (self->video_fd, VIDIOC_S_FMT, &fmt) < 0 ||
      fmt.fmt.pix_mp.width != GST_TOF_WIDTH || fmt.fmt.pix_mp.height != GST_TOF_HEIGHT ||
      fmt.fmt.pix_mp.plane_fmt[0].bytesperline < RAW_LINE_BYTES) {
    GST_ELEMENT_ERROR (self, RESOURCE, SETTINGS, ("Cannot set 240x180 RAW12 on %s.",
            self->video_path), ("%s", g_strerror (errno)));
    goto fail;
  }
  self->stride = fmt.fmt.pix_mp.plane_fmt[0].bytesperline;

  memset (&req, 0, sizeof (req));
  req.count = self->n_v4l2_buffers;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  req.memory = V4L2_MEMORY_MMAP;
  if (xioctl (self->video_fd, VIDIOC_REQBUFS, &req) < 0 || req.count < 4) {
    GST_ELEMENT_ERROR (self, RESOURCE, NO_SPACE_LEFT, ("Cannot get V4L2 buffers."),
        ("%s (count %u)", g_strerror (errno), req.count));
    goto fail;
  }
  self->bufs = g_new0 (TofMmap, req.count);
  self->n_bufs = req.count;
  for (guint i = 0; i < self->n_bufs; i++) {
    struct v4l2_plane pl[1];
    struct v4l2_buffer b;

    memset (pl, 0, sizeof (pl));
    memset (&b, 0, sizeof (b));
    b.type = req.type;
    b.memory = V4L2_MEMORY_MMAP;
    b.index = i;
    b.m.planes = pl;
    b.length = 1;
    if (xioctl (self->video_fd, VIDIOC_QUERYBUF, &b) < 0)
      goto buf_fail;
    self->bufs[i].length = pl[0].length;
    self->bufs[i].start = mmap (NULL, pl[0].length, PROT_READ, MAP_SHARED,
        self->video_fd, pl[0].m.mem_offset);
    if (self->bufs[i].start == MAP_FAILED)
      goto buf_fail;
    if (xioctl (self->video_fd, VIDIOC_QBUF, &b) < 0)
      goto buf_fail;
  }

  gst_poll_fd_init (&self->pollfd);
  self->pollfd.fd = self->video_fd;
  gst_poll_add_fd (self->poll, &self->pollfd);
  gst_poll_fd_ctl_read (self->poll, &self->pollfd, TRUE);
  gst_poll_set_flushing (self->poll, FALSE);

  self->count = 0;
  self->have_last = FALSE;
  self->group_seq = 0;
  self->last_frame_wall = monotonic_ns ();
  GST_OBJECT_LOCK (self);
  self->n_pushed = self->n_incomplete = self->n_stray = 0;
  self->n_seq_gaps = self->n_switch_dropped = 0;
  GST_OBJECT_UNLOCK (self);
  GST_INFO_OBJECT (self, "ready on %s (stride %u, %u buffers)", self->video_path,
      self->stride, self->n_bufs);
  return TRUE;

no_entity:
  close (media_fd);
  GST_ELEMENT_ERROR (self, RESOURCE, NOT_FOUND, ("Entity %s not in the CAMSS graph.", name),
      (NULL));
  return FALSE;
buf_fail:
  GST_ELEMENT_ERROR (self, RESOURCE, NO_SPACE_LEFT, ("Cannot map V4L2 buffers."),
      ("%s", g_strerror (errno)));
fail:
  release_device (self);
  return FALSE;
}

static gboolean
gst_tof_src_stop (GstBaseSrc * basesrc)
{
  GstTofSrc *self = GST_TOF_SRC (basesrc);

  release_device (self);
  GST_INFO_OBJECT (self, "stopped: %" G_GUINT64_FORMAT " groups pushed, %"
      G_GUINT64_FORMAT " incomplete", self->n_pushed, self->n_incomplete);
  return TRUE;
}

static gboolean
gst_tof_src_unlock (GstBaseSrc * basesrc)
{
  gst_poll_set_flushing (GST_TOF_SRC (basesrc)->poll, TRUE);
  return TRUE;
}

static gboolean
gst_tof_src_unlock_stop (GstBaseSrc * basesrc)
{
  gst_poll_set_flushing (GST_TOF_SRC (basesrc)->poll, FALSE);
  return TRUE;
}

static gboolean
gst_tof_src_query (GstBaseSrc * basesrc, GstQuery * query)
{
  GstTofSrc *self = GST_TOF_SRC (basesrc);

  if (GST_QUERY_TYPE (query) == GST_QUERY_LATENCY) {
    GstClockTime min = GROUP_LATENCY;
    GstClockTime max = min + self->n_v4l2_buffers * (GST_SECOND / 120);

    gst_query_set_latency (query, TRUE, min, max);
    return TRUE;
  }
  return GST_BASE_SRC_CLASS (gst_tof_src_parent_class)->query (basesrc, query);
}

/* RAW12 MIPI packing (2 pixels in 3 bytes) -> int16: the module's samples are 11-bit two's complement */
static void
unpack_frame (const guint8 * src, guint stride, guint8 * dst)
{
  for (guint y = 0; y < GST_TOF_HEIGHT; y++) {
    const guint8 *s = src + y * stride;
    gint16 *d = (gint16 *) (dst + y * GST_TOF_PLANE_STRIDE);

    for (guint x = 0; x < GST_TOF_WIDTH / 2; x++, s += 3) {
      gint p0 = (s[0] << 4) | (s[2] & 0xf);
      gint p1 = (s[1] << 4) | (s[2] >> 4);

      d[2 * x] = GINT16_TO_LE ((p0 & 0x400) ? p0 - 2048 : p0);
      d[2 * x + 1] = GINT16_TO_LE ((p1 & 0x400) ? p1 - 2048 : p1);
    }
  }
}

static void
discard_pending (GstTofSrc * self)
{
  if (self->pending) {
    gst_buffer_unmap (self->pending, &self->pmap);
    gst_clear_buffer (&self->pending);
  }
  self->count = 0;
}

static GstClockTime
running_time_of (GstTofSrc * self, guint64 mono_ns)
{
  GstClock *clock = gst_element_get_clock (GST_ELEMENT (self));
  GstClockTime base, now, t = GST_CLOCK_TIME_NONE;
  guint64 age;

  if (!clock)
    return GST_CLOCK_TIME_NONE;
  base = gst_element_get_base_time (GST_ELEMENT (self));
  now = gst_clock_get_time (clock);
  age = monotonic_ns () - mono_ns;
  if (now > age && now - age > base)
    t = now - age - base;
  else
    t = 0;
  gst_object_unref (clock);
  return t;
}

static GstFlowReturn
gst_tof_src_create (GstPushSrc * pushsrc, GstBuffer ** out)
{
  GstTofSrc *self = GST_TOF_SRC (pushsrc);

  if (!self->streaming) {
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;

    /* first create(): stream only once data is wanted, so no stale frames pile up in PAUSED */
    if (xioctl (self->video_fd, VIDIOC_STREAMON, &type) < 0) {
      GST_ELEMENT_ERROR (self, RESOURCE, FAILED, ("STREAMON failed on %s.", self->video_path),
          ("%s", g_strerror (errno)));
      return GST_FLOW_ERROR;
    }
    self->streaming = TRUE;
    self->last_frame_wall = monotonic_ns ();
    /* the module starts in its default (far) mode and switches like after a range write,
     * so the groups right after STREAMON are dropped the same way */
    GST_OBJECT_LOCK (self);
    self->drop_left = STREAMON_DROP_GROUPS;
    GST_OBJECT_UNLOCK (self);
    GST_INFO_OBJECT (self, "STREAMON %s", self->video_path);
  }

  while (TRUE) {
    struct v4l2_plane pl[1];
    struct v4l2_buffer b;
    guint64 ts;
    gboolean gap, seq_break;
    gint r;

    r = gst_poll_wait (self->poll, 200 * GST_MSECOND);
    if (r < 0) {
      if (errno == EBUSY)
        return GST_FLOW_FLUSHING;
      if (errno == EINTR || errno == EAGAIN)
        continue;
      GST_ELEMENT_ERROR (self, RESOURCE, READ, ("poll failed."), ("%s", g_strerror (errno)));
      return GST_FLOW_ERROR;
    }

    memset (pl, 0, sizeof (pl));
    memset (&b, 0, sizeof (b));
    b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    b.memory = V4L2_MEMORY_MMAP;
    b.m.planes = pl;
    b.length = 1;
    if (r == 0 || xioctl (self->video_fd, VIDIOC_DQBUF, &b) < 0) {
      if (r > 0 && errno != EAGAIN) {
        GST_ELEMENT_ERROR (self, RESOURCE, READ, ("DQBUF failed."), ("%s", g_strerror (errno)));
        return GST_FLOW_ERROR;
      }
      if (monotonic_ns () - self->last_frame_wall > NO_FRAME_TIMEOUT) {
        GST_ELEMENT_ERROR (self, RESOURCE, READ, ("No frame from the ToF module for 1 s."),
            ("Check the CSI mux line (mux-gpio), the drivers and the sensor link."));
        return GST_FLOW_ERROR;
      }
      continue;
    }
    self->last_frame_wall = monotonic_ns ();
    ts = GST_TIMEVAL_TO_TIME (b.timestamp);

    gap = self->have_last && ts - self->last_ts > GROUP_GAP_NS;
    seq_break = self->have_last && b.sequence != self->last_seq + 1;
    if (seq_break) {
      GST_OBJECT_LOCK (self);
      self->n_seq_gaps++;
      if (self->count > 0)
        self->n_incomplete++;
      GST_OBJECT_UNLOCK (self);
      discard_pending (self);
    }
    if (gap && self->count > 0) {
      GST_OBJECT_LOCK (self);
      self->n_incomplete++;
      GST_OBJECT_UNLOCK (self);
      discard_pending (self);
    }
    self->last_ts = ts;
    self->last_seq = b.sequence;
    self->have_last = TRUE;

    if (self->count == 0 && !gap) {
      /* not after a gap: frames before the first gap after STREAMON, or a 5th frame */
      GST_OBJECT_LOCK (self);
      self->n_stray++;
      GST_OBJECT_UNLOCK (self);
    } else {
      if (self->count == 0) {
        self->pending = gst_buffer_new_allocate (NULL, GST_TOF_BUFFER_SIZE, NULL);
        gst_buffer_map (self->pending, &self->pmap, GST_MAP_WRITE);
      }
      if (pl[0].bytesused && pl[0].bytesused < self->stride * GST_TOF_HEIGHT)
        GST_WARNING_OBJECT (self, "short frame: %u bytes", pl[0].bytesused);
      unpack_frame (self->bufs[b.index].start, self->stride,
          self->pmap.data + GST_TOF_HEADER_SIZE + self->count * GST_TOF_PLANE_SIZE);
      self->seqs[self->count] = b.sequence;
      self->ts[self->count] = ts;
      self->count++;
    }

    if (xioctl (self->video_fd, VIDIOC_QBUF, &b) < 0) {
      GST_ELEMENT_ERROR (self, RESOURCE, READ, ("QBUF failed."), ("%s", g_strerror (errno)));
      return GST_FLOW_ERROR;
    }

    if (self->count == GST_TOF_PHASES) {
      guint8 *h = self->pmap.data;
      guint16 flags = 0;
      guint8 mode, req;
      gboolean drop = FALSE;

      self->group_seq++;
      GST_OBJECT_LOCK (self);
      if (self->drop_left > 0) {
        self->drop_left--;
        drop = TRUE;
      } else if (self->ts[0] < self->drop_before_ts) {
        self->n_switch_dropped++;
        drop = TRUE;
      } else {
        if (self->after_switch)
          flags |= GST_TOF_FLAG_AFTER_SWITCH;
        self->after_switch = FALSE;
        self->n_pushed++;
      }
      mode = self->applied_range;
      req = self->range;
      if (!drop && self->in_reference && mode == GST_TOF_RANGE_FAR) {
        /* the reference: back to the requested mode, also if the interval was set to 0 meanwhile */
        flags |= GST_TOF_FLAG_REFERENCE;
        self->n_references++;
        apply_range (self);
      } else if (!drop && self->far_ref_interval > 0 && self->range == GST_TOF_RANGE_NEAR &&
          mode == GST_TOF_RANGE_NEAR && !self->in_reference &&
          ++self->ref_near_count >= self->far_ref_interval) {
        self->in_reference = TRUE;
        apply_mode (self, GST_TOF_RANGE_FAR);
      }
      GST_OBJECT_UNLOCK (self);
      if (drop) {
        discard_pending (self);
        continue;
      }

      memset (h, 0, GST_TOF_HEADER_SIZE);
      GST_WRITE_UINT32_LE (h + GST_TOF_OFF_MAGIC, GST_TOF_MAGIC);
      GST_WRITE_UINT16_LE (h + GST_TOF_OFF_HDR_SIZE, GST_TOF_HEADER_SIZE);
      GST_WRITE_UINT16_LE (h + GST_TOF_OFF_FLAGS, flags);
      GST_WRITE_UINT32_LE (h + GST_TOF_OFF_GROUP_SEQ, self->group_seq);
      for (guint k = 0; k < GST_TOF_PHASES; k++)
        GST_WRITE_UINT32_LE (h + GST_TOF_OFF_FRAME_SEQ + 4 * k, self->seqs[k]);
      GST_WRITE_UINT64_LE (h + GST_TOF_OFF_TIMESTAMP, self->ts[0]);
      GST_WRITE_UINT32_LE (h + GST_TOF_OFF_DURATION, (guint32) (self->ts[3] - self->ts[0]));
      GST_WRITE_UINT8 (h + GST_TOF_OFF_MODE, mode);
      GST_WRITE_UINT8 (h + GST_TOF_OFF_REQ_MODE, req);
      GST_WRITE_UINT32_LE (h + GST_TOF_OFF_FIRMWARE, self->firmware_version);
      gst_buffer_unmap (self->pending, &self->pmap);

      GST_BUFFER_PTS (self->pending) = running_time_of (self, self->ts[0]);
      GST_BUFFER_DURATION (self->pending) = GROUP_DURATION;
      GST_BUFFER_OFFSET (self->pending) = self->group_seq;
      *out = self->pending;
      self->pending = NULL;
      self->count = 0;
      return GST_FLOW_OK;
    }
  }
}

static void
gst_tof_src_set_property (GObject * object, guint prop_id, const GValue * value,
    GParamSpec * pspec)
{
  GstTofSrc *self = GST_TOF_SRC (object);

  GST_OBJECT_LOCK (self);
  switch (prop_id) {
    case PROP_MEDIA_DEVICE:
      g_free (self->media_device);
      self->media_device = g_value_dup_string (value);
      break;
    case PROP_DEVICE:
      g_free (self->device);
      self->device = g_value_dup_string (value);
      break;
    case PROP_CSIPHY:
      self->csiphy = g_value_get_uint (value);
      break;
    case PROP_SETUP_PIPELINE:
      self->setup_pipeline = g_value_get_boolean (value);
      break;
    case PROP_MUX_GPIO:
      g_free (self->mux_gpio);
      self->mux_gpio = g_value_dup_string (value);
      break;
    case PROP_RANGE:
      self->range = g_value_get_enum (value);
      self->in_reference = FALSE;       /* a range write ends a pending far reference */
      self->ref_near_count = 0;
      if (self->subdev_fd >= 0 && self->range != self->applied_range)
        apply_range (self);
      else if (self->subdev_fd < 0)
        self->applied_range = self->range;
      break;
    case PROP_EXPOSURE:
      self->exposure = g_value_get_int (value);
      self->exposure_set = TRUE;
      if (write_ctrl (self, V4L2_CID_EXPOSURE, self->exposure, "exposure"))
        warn_ignored (self, "exposure");
      break;
    case PROP_FRAME_RATE:
      self->frame_rate = g_value_get_int (value);
      self->frame_rate_set = TRUE;
      if (write_ctrl (self, CID_FRAME_RATE, self->frame_rate, "frame_rate"))
        warn_ignored (self, "frame-rate");
      break;
    case PROP_NUM_V4L2_BUFFERS:
      self->n_v4l2_buffers = g_value_get_uint (value);
      break;
    case PROP_SETTLE_TIME:
      self->settle_ms = g_value_get_uint (value);
      break;
    case PROP_FAR_REFERENCE_INTERVAL:
      self->far_ref_interval = g_value_get_uint (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
  GST_OBJECT_UNLOCK (self);
}

static void
gst_tof_src_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec)
{
  GstTofSrc *self = GST_TOF_SRC (object);
  gint32 v;

  GST_OBJECT_LOCK (self);
  switch (prop_id) {
    case PROP_MEDIA_DEVICE:
      g_value_set_string (value, self->media_device);
      break;
    case PROP_DEVICE:
      g_value_set_string (value, self->device ? self->device : self->video_path);
      break;
    case PROP_CSIPHY:
      g_value_set_uint (value, self->csiphy);
      break;
    case PROP_SETUP_PIPELINE:
      g_value_set_boolean (value, self->setup_pipeline);
      break;
    case PROP_MUX_GPIO:
      g_value_set_string (value, self->mux_gpio);
      break;
    case PROP_RANGE:
      /* read back from the driver while open (the module MCU itself cannot report it) */
      g_value_set_enum (value, read_ctrl (self->subdev_fd, CID_TRIGGER_MODE, &v) ?
          (v ? GST_TOF_RANGE_NEAR : GST_TOF_RANGE_FAR) : self->range);
      break;
    case PROP_EXPOSURE:
      g_value_set_int (value, read_ctrl (self->subdev_fd, V4L2_CID_EXPOSURE, &v) ?
          v : self->exposure);
      break;
    case PROP_FRAME_RATE:
      g_value_set_int (value, read_ctrl (self->subdev_fd, CID_FRAME_RATE, &v) ?
          v : self->frame_rate);
      break;
    case PROP_FIRMWARE_VERSION:
      g_value_set_uint (value, self->firmware_version);
      break;
    case PROP_DEVICE_ID:
      g_value_set_uint (value, self->device_id);
      break;
    case PROP_NUM_V4L2_BUFFERS:
      g_value_set_uint (value, self->n_v4l2_buffers);
      break;
    case PROP_SETTLE_TIME:
      g_value_set_uint (value, self->settle_ms);
      break;
    case PROP_FAR_REFERENCE_INTERVAL:
      g_value_set_uint (value, self->far_ref_interval);
      break;
    case PROP_STATS:
      g_value_take_boxed (value, gst_structure_new ("stats",
              "groups-pushed", G_TYPE_UINT64, self->n_pushed,
              "groups-incomplete", G_TYPE_UINT64, self->n_incomplete,
              "frames-stray", G_TYPE_UINT64, self->n_stray,
              "sequence-gaps", G_TYPE_UINT64, self->n_seq_gaps,
              "groups-dropped-switch", G_TYPE_UINT64, self->n_switch_dropped,
              "groups-dropped-start", G_TYPE_UINT64, (guint64) (STREAMON_DROP_GROUPS - self->drop_left),
              "range-switches", G_TYPE_UINT64, self->n_switches,
              "far-references", G_TYPE_UINT64, self->n_references, NULL));
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
  GST_OBJECT_UNLOCK (self);
}

static void
gst_tof_src_finalize (GObject * object)
{
  GstTofSrc *self = GST_TOF_SRC (object);

  g_free (self->media_device);
  g_free (self->device);
  g_free (self->mux_gpio);
  gst_poll_free (self->poll);
  G_OBJECT_CLASS (gst_tof_src_parent_class)->finalize (object);
}

static void
gst_tof_src_class_init (GstTofSrcClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *element_class = GST_ELEMENT_CLASS (klass);
  GstBaseSrcClass *basesrc_class = GST_BASE_SRC_CLASS (klass);
  GstPushSrcClass *pushsrc_class = GST_PUSH_SRC_CLASS (klass);
  const GParamFlags rw = G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS;
  const GParamFlags ro = G_PARAM_READABLE | G_PARAM_STATIC_STRINGS;

  gobject_class->set_property = gst_tof_src_set_property;
  gobject_class->get_property = gst_tof_src_get_property;
  gobject_class->finalize = gst_tof_src_finalize;

  g_object_class_install_property (gobject_class, PROP_MEDIA_DEVICE,
      g_param_spec_string ("media-device", "Media device",
          "Media controller node of CAMSS (NULL = the one whose driver is qcom-camss)",
          NULL, rw | GST_PARAM_MUTABLE_READY));
  g_object_class_install_property (gobject_class, PROP_DEVICE,
      g_param_spec_string ("device", "Device",
          "RDI video node (NULL = msm_vfe<csiphy>_video0 of the media graph)",
          NULL, rw | GST_PARAM_MUTABLE_READY));
  g_object_class_install_property (gobject_class, PROP_CSIPHY,
      g_param_spec_uint ("csiphy", "CSIPHY",
          "CSI path msm_csiphyN -> msm_csidN -> msm_vfeN_rdi0",
          0, 3, DEFAULT_CSIPHY, rw | GST_PARAM_MUTABLE_READY));
  g_object_class_install_property (gobject_class, PROP_SETUP_PIPELINE,
      g_param_spec_boolean ("setup-pipeline", "Set up pipeline",
          "Enable the CAMSS links and set 240x180 SRGGB12_1X12 on every pad at start",
          DEFAULT_SETUP, rw | GST_PARAM_MUTABLE_READY));
  g_object_class_install_property (gobject_class, PROP_MUX_GPIO,
      g_param_spec_string ("mux-gpio", "Mux GPIO",
          "CSI mux line held while streaming, gpiochipN:LINE=VALUE; \"auto\" = the line of a known board "
          "for this csiphy (Arduino VENTUNO Q CAMERA0), none elsewhere; empty = untouched",
          DEFAULT_MUX_GPIO, rw | GST_PARAM_MUTABLE_READY));
  g_object_class_install_property (gobject_class, PROP_RANGE,
      g_param_spec_enum ("range", "Range",
          "Range mode (trigger_mode); groups in flight after a change are dropped",
          GST_TYPE_TOF_RANGE, DEFAULT_RANGE, rw | GST_PARAM_MUTABLE_PLAYING));
  g_object_class_install_property (gobject_class, PROP_EXPOSURE,
      g_param_spec_int ("exposure", "Exposure",
          "Module exposure control (ignored by module firmware 0x10002)",
          1, 65523, DEFAULT_EXPOSURE, rw | GST_PARAM_MUTABLE_PLAYING));
  g_object_class_install_property (gobject_class, PROP_FRAME_RATE,
      g_param_spec_int ("frame-rate", "Frame rate",
          "Module frame-rate control (ignored by module firmware 0x10002: fixed 120 frames/s)",
          5, 120, DEFAULT_FRAME_RATE, rw | GST_PARAM_MUTABLE_PLAYING));
  g_object_class_install_property (gobject_class, PROP_FIRMWARE_VERSION,
      g_param_spec_uint ("firmware-version", "Firmware version",
          "Module MCU firmware version (0 = unknown; valid after start)", 0, G_MAXUINT32, 0, ro));
  g_object_class_install_property (gobject_class, PROP_DEVICE_ID,
      g_param_spec_uint ("device-id", "Device ID",
          "Module MCU device ID (0 = unknown; valid after start)", 0, G_MAXUINT32, 0, ro));
  g_object_class_install_property (gobject_class, PROP_NUM_V4L2_BUFFERS,
      g_param_spec_uint ("num-v4l2-buffers", "V4L2 buffers", "V4L2 MMAP buffers",
          4, 32, DEFAULT_V4L2_BUFFERS, rw | GST_PARAM_MUTABLE_READY));
  g_object_class_install_property (gobject_class, PROP_SETTLE_TIME,
      g_param_spec_uint ("settle-time", "Settle time (ms)",
          "After a range change, drop groups whose first frame starts sooner than this "
          "(the module switches 40-50 ms after the write)",
          0, 1000, DEFAULT_SETTLE_MS, rw | GST_PARAM_MUTABLE_PLAYING));
  g_object_class_install_property (gobject_class, PROP_FAR_REFERENCE_INTERVAL,
      g_param_spec_uint ("far-reference-interval", "Far reference interval",
          "With range=near, take one far group after this many near groups, flagged as a reference for "
          "qtitofdepth unwrap (each costs the far group and the groups dropped at two range writes); "
          "0 = off", 0, G_MAXUINT, 0, rw | GST_PARAM_MUTABLE_PLAYING));
  g_object_class_install_property (gobject_class, PROP_STATS,
      g_param_spec_boxed ("stats", "Statistics",
          "groups-pushed, groups-incomplete, frames-stray, sequence-gaps, "
          "groups-dropped-switch, range-switches, far-references", GST_TYPE_STRUCTURE, ro));

  gst_element_class_add_static_pad_template (element_class, &src_template);
  gst_element_class_set_static_metadata (element_class, "Arducam ToF source",
      "Source/Video", "Groups of 4 raw DCS frames from the Arducam ToF on the CAMSS "
      "RDI path, with range control", "arducam-tof-ventuno project");

  basesrc_class->start = GST_DEBUG_FUNCPTR (gst_tof_src_start);
  basesrc_class->stop = GST_DEBUG_FUNCPTR (gst_tof_src_stop);
  basesrc_class->unlock = GST_DEBUG_FUNCPTR (gst_tof_src_unlock);
  basesrc_class->unlock_stop = GST_DEBUG_FUNCPTR (gst_tof_src_unlock_stop);
  basesrc_class->query = GST_DEBUG_FUNCPTR (gst_tof_src_query);
  pushsrc_class->create = GST_DEBUG_FUNCPTR (gst_tof_src_create);

  GST_DEBUG_CATEGORY_INIT (gst_tof_src_debug, "qtitofsrc", 0, "Arducam ToF source");
}

static void
gst_tof_src_init (GstTofSrc * self)
{
  self->csiphy = DEFAULT_CSIPHY;
  self->setup_pipeline = DEFAULT_SETUP;
  self->mux_gpio = g_strdup (DEFAULT_MUX_GPIO);
  self->range = DEFAULT_RANGE;
  self->exposure = DEFAULT_EXPOSURE;
  self->frame_rate = DEFAULT_FRAME_RATE;
  self->n_v4l2_buffers = DEFAULT_V4L2_BUFFERS;
  self->settle_ms = DEFAULT_SETTLE_MS;
  self->video_fd = self->subdev_fd = self->gpio_fd = -1;
  self->applied_range = DEFAULT_RANGE;
  self->poll = gst_poll_new (TRUE);
  gst_poll_fd_init (&self->pollfd);

  gst_base_src_set_live (GST_BASE_SRC (self), TRUE);
  gst_base_src_set_format (GST_BASE_SRC (self), GST_FORMAT_TIME);
  gst_base_src_set_do_timestamp (GST_BASE_SRC (self), FALSE);
}
