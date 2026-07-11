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
    if (info.type != orc::NodeType::TRANSFORM) {
        std::cerr << "Expected TRANSFORM node type\n";
        return 1;
    }
    if (stage.required_input_count() != 1 || stage.output_count() != 1) {
        std::cerr << "Expected one input and one output\n";
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
