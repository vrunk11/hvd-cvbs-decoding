{
  description = "Decode-Orc HVD chroma decoder stage plugin";

  # Keep the nixpkgs input aligned with the decode-orc host repository's
  # flake (github:simoninns/decode-orc). Since Decode-Orc 2.0 (host ABI 5)
  # the plugin loader requires the plugin's toolchain tag — compiler family,
  # major version, and C++ standard library — to equal the host's exactly.
  # Building from a different nixpkgs generation (and therefore a different
  # default gcc) produces plugins the host refuses to load.
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };

        # Use the nixpkgs default stdenv (gcc + libstdc++), exactly as the
        # decode-orc flake does. Do not add clang here: compiling the plugin
        # with a different compiler family than the host changes the
        # toolchain tag and the host will reject the binary.
        stdenv = pkgs.stdenv;
      in {
        devShells.default = pkgs.mkShell.override { inherit stdenv; } {
          packages = with pkgs; [
            cmake
            ninja
            pkg-config

            # Third-party headers permitted by the plugin SDK allowlist.
            fmt
            spdlog

            # Link-interface dependencies of orc::orc-core. The SDK's CMake
            # config (decode-orc-plugin-sdkConfig.cmake) resolves these with
            # find_dependency()/pkg_check_modules() before importing its
            # targets, so configure fails without them even though the plugin
            # code never includes their headers.
            sqlite
            yaml-cpp
            libpng
            ffmpeg

            # HVD engine dependency (single-precision FFT).
            fftw

            # Development tools (mirrors the decode-orc dev shell subset
            # relevant to plugin work).
            clang-tools
            ccache
            git
          ] ++ pkgs.lib.optionals pkgs.stdenv.isLinux [
            gdb
          ];

          shellHook = ''
            echo "orc-plugin_hvd nix development environment"
            echo "C++ toolchain: $(g++ --version | head -n1)"

            # Set up ccache if available
            export CMAKE_CXX_COMPILER_LAUNCHER=ccache
          '';

          CMAKE_EXPORT_COMPILE_COMMANDS = 1;
          # Default to Ninja when no -G is given, matching decode-orc.
          CMAKE_GENERATOR = "Ninja";
        };
      }
    );
}
