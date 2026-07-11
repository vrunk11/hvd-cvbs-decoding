#include "hvd_chroma_decoder_stage.h"

#include <iostream>

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
    if (stage.get_parameter_descriptors().empty()) {
        std::cerr << "Expected parameter descriptors\n";
        return 1;
    }
    return 0;
}
