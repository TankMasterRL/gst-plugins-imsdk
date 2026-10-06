/*
 * Copyright (c) 2026, arducam-tof-ventuno project.
 * SPDX-License-Identifier: BSD-3-Clause-Clear
 */

#ifndef __GST_TOF_DEPTH_H__
#define __GST_TOF_DEPTH_H__

#include <gst/gst.h>

G_BEGIN_DECLS

#define GST_TYPE_TOF_DEPTH (gst_tof_depth_get_type ())
G_DECLARE_FINAL_TYPE (GstTofDepth, gst_tof_depth, GST, TOF_DEPTH, GstElement)

G_END_DECLS

#endif /* __GST_TOF_DEPTH_H__ */
