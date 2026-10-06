/*
 * Copyright (c) 2026, arducam-tof-ventuno project.
 * SPDX-License-Identifier: BSD-3-Clause-Clear
 *
 * DCS16 group buffers exchanged between qtitofsrc and qtitofdepth (DEC-0020, wiki/ToF-Elements.md):
 * a 64-byte little-endian header followed by four 240x180 planes of int16 (F0..F3, the module's
 * 11-bit two's-complement samples sign-extended, row stride 480 bytes).
 */

#ifndef __GST_TOF_DCS_H__
#define __GST_TOF_DCS_H__

#include <gst/gst.h>

G_BEGIN_DECLS

#define GST_TOF_DCS_CAPS \
  "video/x-tof-dcs, format=(string)DCS16, width=(int)240, height=(int)180, " \
  "phases=(int)4, framerate=(fraction)30/1"

#define GST_TOF_WIDTH          240
#define GST_TOF_HEIGHT         180
#define GST_TOF_PHASES         4
#define GST_TOF_PLANE_STRIDE   (GST_TOF_WIDTH * 2)
#define GST_TOF_PLANE_SIZE     (GST_TOF_PLANE_STRIDE * GST_TOF_HEIGHT)
#define GST_TOF_HEADER_SIZE    64
#define GST_TOF_BUFFER_SIZE    (GST_TOF_HEADER_SIZE + GST_TOF_PHASES * GST_TOF_PLANE_SIZE)

#define GST_TOF_MAGIC          0x31464f54 /* "TOF1" */

/* header flags */
#define GST_TOF_FLAG_MODE_DETECTED   (1 << 0) /* mode measured from the data; else assumed */
#define GST_TOF_FLAG_AFTER_SWITCH    (1 << 1) /* first group pushed after a range write */

/* header field offsets */
#define GST_TOF_OFF_MAGIC      0
#define GST_TOF_OFF_HDR_SIZE   4
#define GST_TOF_OFF_FLAGS      6
#define GST_TOF_OFF_GROUP_SEQ  8
#define GST_TOF_OFF_FRAME_SEQ  12  /* 4 x u32 */
#define GST_TOF_OFF_TIMESTAMP  28  /* u64 ns, CLOCK_MONOTONIC, end of F0 */
#define GST_TOF_OFF_DURATION   36  /* u32 ns, F0 -> F3 */
#define GST_TOF_OFF_MODE       40  /* u8: 0 far (R 4 m), 1 near (R 2 m) */
#define GST_TOF_OFF_REQ_MODE   41  /* u8: trigger_mode requested */
#define GST_TOF_OFF_FIRMWARE   42  /* u32 */

typedef enum {
  GST_TOF_RANGE_FAR = 0,
  GST_TOF_RANGE_NEAR = 1,
} GstTofRange;

GType gst_tof_range_get_type (void);
#define GST_TYPE_TOF_RANGE (gst_tof_range_get_type ())

G_END_DECLS

#endif /* __GST_TOF_DCS_H__ */
