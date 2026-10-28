/*
 * Copyright (c) 2026, arducam-tof-ventuno project.
 * SPDX-License-Identifier: BSD-3-Clause-Clear
 *
 * Plugin qtitof: elements for the Arducam ToF camera on the CAMSS RDI path.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "tofdcs.h"
#include "tofsrc.h"
#include "tofdepth.h"

GType
gst_tof_range_get_type (void)
{
  static gsize type = 0;
  static const GEnumValue values[] = {
    { GST_TOF_RANGE_FAR, "Far mode, 4 m unambiguous range (trigger_mode 0)", "far" },
    { GST_TOF_RANGE_NEAR, "Near mode, 2 m unambiguous range (trigger_mode 1)", "near" },
    { 0, NULL, NULL },
  };

  if (g_once_init_enter (&type)) {
    GType t = g_enum_register_static ("GstTofRange", values);
    g_once_init_leave (&type, t);
  }
  return type;
}

static gboolean
plugin_init (GstPlugin * plugin)
{
  return gst_element_register (plugin, "qtitofsrc", GST_RANK_NONE,
      GST_TYPE_TOF_SRC) &&
      gst_element_register (plugin, "qtitofdepth", GST_RANK_NONE,
      GST_TYPE_TOF_DEPTH);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    qtitof,
    "Arducam ToF camera on the CAMSS RDI path (source and depth)",
    plugin_init,
    PACKAGE_VERSION, PACKAGE_LICENSE, PACKAGE_SUMMARY, PACKAGE_ORIGIN)
