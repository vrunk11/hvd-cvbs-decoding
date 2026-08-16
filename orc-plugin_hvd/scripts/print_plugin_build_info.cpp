/*
 * File:        print_plugin_build_info.cpp
 * Module:      orc-stage-plugin-hvd-chroma-decoder (release tooling)
 * Purpose:     Print plugin_id / host_abi / toolchain_tag exactly as they are
 *              embedded in kPluginDescriptor for THIS build — used by CI to
 *              generate orc-plugin-manifest.yaml without re-deriving the ABI
 *              number or toolchain tag by hand anywhere else. Reading them
 *              off the actual compiled descriptor means the manifest can
 *              never drift from the binary it describes.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2026 hvd-decode contributors
 *
 * Not part of the plugin itself: built as a standalone tiny executable,
 * linked only against the SDK headers (no hvd_core, no FFTW). See the
 * `hvd_print_build_info` target in ../CMakeLists.txt.
 */

#include <cstdio>

#include "plugin.h"

int main() {
  const orc::StagePluginDescriptor& d = orc::plugins::hvd::kPluginDescriptor;
  std::printf("plugin_id=%s\n", d.plugin_id);
  std::printf("host_abi=%u\n", d.host_abi_version);
  std::printf("toolchain_tag=%s\n", d.toolchain_tag);
  return 0;
}
