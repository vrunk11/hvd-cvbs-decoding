#include "hvd_chroma_decoder_stage.h"

#include <cmath>
#include <iostream>
#include <string>
#include <variant>

int main()
{
    orc::plugins::hvd::HvdChromaDecoderStage stage;

    const auto info = stage.get_node_type_info();
    if (info.stage_name != "hvd_chroma_decoder") {
        std::cerr << "Unexpected stage_name: " << info.stage_name << '\n';
        return 1;
    }
    // Sink now: colour preview + direct file export, no downstream Y/C
    // representation (see docs/PORTING.md and the split-field/video-sink
    // writeup for why the transform output was removed).
    if (info.type != orc::NodeType::SINK) {
        std::cerr << "Expected SINK node type\n";
        return 1;
    }
    if (stage.required_input_count() != 1 || stage.output_count() != 0) {
        std::cerr << "Expected one input and zero outputs\n";
        return 1;
    }
    // Preview capability must be invalid before execute() has cached data.
    if (stage.get_preview_capability().is_valid()) {
        std::cerr << "Preview capability must be invalid before execute()\n";
        return 1;
    }
    // Parameter round-trip: descriptors exist and defaults read back.
    const auto descriptors = stage.get_parameter_descriptors();
    if (descriptors.empty()) {
        std::cerr << "Expected parameter descriptors\n";
        return 1;
    }

    // ------------------------------------------------------------------
    // DESCRIPTOR DEFAULTS MUST EQUAL THE HvdConfig DEFAULTS.
    //
    // There are two sources of truth for every default: the C++ struct
    // (HvdConfig, what the engine and the tests use) and the GUI
    // ParameterDescriptor (what the host actually hands the stage at
    // runtime). When they disagree the descriptor silently wins in the
    // application while every unit test exercises the struct value --
    // which is exactly how `chroma_phase_deg` kept a default of 180 in the
    // GUI after the struct had been corrected to 0, and how
    // `subcarrier_khz` shipped the rounded 2556.8 against the struct's
    // exact 2556.8182. `stage` here is freshly constructed, so
    // get_parameters() returns the struct defaults verbatim.
    // ------------------------------------------------------------------
    const auto values = stage.get_parameters();
    int mismatches = 0;
    for (const auto& d : descriptors) {
        if (!d.constraints.default_value.has_value()) continue;
        const auto it = values.find(d.name);
        if (it == values.end()) continue;  // preview-only knobs, no config field
        const orc::ParameterValue& want = *d.constraints.default_value;
        const orc::ParameterValue& got = it->second;
        bool equal = false;
        if (std::holds_alternative<double>(want) &&
            std::holds_alternative<double>(got)) {
            // HvdConfig stores these numeric defaults as float. Compare both
            // values in float so the descriptor and HvdConfig use the same
            // representation instead of comparing float-vs-double rounding.
            const float a = static_cast<float>(std::get<double>(want));
            const float b = static_cast<float>(std::get<double>(got));
            equal = a == b;
            if (!equal) {
                std::cerr << "Default mismatch for '" << d.name
                          << "': descriptor=" << a << " HvdConfig=" << b << '\n';
            }
        } else if (std::holds_alternative<bool>(want) &&
                   std::holds_alternative<bool>(got)) {
            equal = std::get<bool>(want) == std::get<bool>(got);
            if (!equal) {
                std::cerr << "Default mismatch for '" << d.name
                          << "': descriptor=" << std::get<bool>(want)
                          << " HvdConfig=" << std::get<bool>(got) << '\n';
            }
        } else if (std::holds_alternative<int32_t>(want) &&
                   std::holds_alternative<int32_t>(got)) {
            equal = std::get<int32_t>(want) == std::get<int32_t>(got);
            if (!equal) {
                std::cerr << "Default mismatch for '" << d.name
                          << "': descriptor=" << std::get<int32_t>(want)
                          << " HvdConfig=" << std::get<int32_t>(got) << '\n';
            }
        } else if (std::holds_alternative<std::string>(want) &&
                   std::holds_alternative<std::string>(got)) {
            equal = std::get<std::string>(want) == std::get<std::string>(got);
            if (!equal) {
                std::cerr << "Default mismatch for '" << d.name << "'\n";
            }
        } else {
            std::cerr << "Type mismatch between descriptor and value for '"
                      << d.name << "'\n";
        }
        if (!equal) ++mismatches;
    }
    if (mismatches != 0) {
        std::cerr << mismatches << " descriptor/HvdConfig default mismatch(es)\n";
        return 1;
    }

    // The chroma phase trim is a RELATIVE correction on top of the measured
    // burst phase, so a healthy source needs none: pin the default at 0 so
    // no future "known-good starting point" can be baked in again.
    for (const auto& d : descriptors) {
        if (d.name != std::string("chroma_phase_deg")) continue;
        if (!d.constraints.default_value.has_value() ||
            !std::holds_alternative<double>(*d.constraints.default_value) ||
            std::get<double>(*d.constraints.default_value) != 0.0) {
            std::cerr << "chroma_phase_deg default must be 0 (relative trim)\n";
            return 1;
        }
    }
    return 0;
}
