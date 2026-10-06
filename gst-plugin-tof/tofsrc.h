/*
 * Copyright (c) 2026, arducam-tof-ventuno project.
 * SPDX-License-Identifier: BSD-3-Clause-Clear
 */

#ifndef __GST_TOF_SRC_H__
#define __GST_TOF_SRC_H__

#include <gst/gst.h>
#include <gst/base/gstpushsrc.h>

#include "tofdcs.h"

G_BEGIN_DECLS

#define GST_TYPE_TOF_SRC (gst_tof_src_get_type ())
G_DECLARE_FINAL_TYPE (GstTofSrc, gst_tof_src, GST, TOF_SRC, GstPushSrc)

G_END_DECLS

#endif /* __GST_TOF_SRC_H__ */
