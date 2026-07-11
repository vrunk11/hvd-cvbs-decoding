#include <orc/plugin/orc_plugin_sdk.h>

#include <iostream>
#include <string>

ORC_STAGE_PLUGIN_EXPORT const orc::StagePluginDescriptor* orc_get_stage_plugin_descriptor();
ORC_STAGE_PLUGIN_EXPORT bool orc_register_stage_plugin(
    const orc::OrcPluginServices* services,
    void* context,
    bool (*register_stage)(void* context, const char* stage_name, orc::OrcStageFactoryFn factory),
    const char** error_message);

namespace {

bool register_probe(void* context, const char* stage_name, orc::OrcStageFactoryFn)
{
    auto* called = static_cast<bool*>(context);
    if (called) {
        *called = true;
    }
    return std::string(stage_name) == "hvd_chroma_decoder";
}

} // namespace

int main()
{
    const auto* descriptor = orc_get_stage_plugin_descriptor();
    if (!descriptor) {
        std::cerr << "Descriptor entrypoint returned null\n";
        return 1;
    }

    if (std::string(descriptor->plugin_id) != "org.decodeorc.stage.hvd_chroma_decoder") {
        std::cerr << "Unexpected plugin_id\n";
        return 1;
    }

    if (descriptor->host_abi_version != orc::kStagePluginHostAbiVersion) {
        std::cerr << "Host ABI mismatch\n";
        return 1;
    }

    if (descriptor->plugin_api_version != orc::kStagePluginApiVersion) {
        std::cerr << "Plugin API version mismatch\n";
        return 1;
    }

    // ABI v5: the loader requires the descriptor's toolchain tag to equal the
    // host's tag exactly. This test binary is built with the same toolchain
    // as the plugin objects, so the tags must match here too.
    if (!descriptor->toolchain_tag ||
        std::string(descriptor->toolchain_tag) != ORC_SDK_TOOLCHAIN_TAG) {
        std::cerr << "Toolchain tag mismatch: "
                  << (descriptor->toolchain_tag ? descriptor->toolchain_tag : "<null>")
                  << " != " << ORC_SDK_TOOLCHAIN_TAG << '\n';
        return 1;
    }

    bool called = false;
    const char* error_message = nullptr;
    if (!orc_register_stage_plugin(nullptr, &called, &register_probe, &error_message)) {
        std::cerr << "Register entrypoint failed: "
                  << (error_message ? error_message : "<none>") << '\n';
        return 1;
    }

    if (!called) {
        std::cerr << "Register callback was not called\n";
        return 1;
    }

    return 0;
}
