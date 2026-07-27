{
  description = "Decode-Orc HVD chroma decoder stage plugin";

  # Keep the nixpkgs input aligned with the decode-orc host repository's
  # flake (github:simoninns/decode-orc). Since Decode-Orc 2.0 (host ABI 5+)
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

        # Use the nixpkgs default stdenv, exactly as the decode-orc flake
        # does: gcc + libstdc++ on Linux, clang + libc++ on macOS. Do not
        # override the compiler family here. Building the plugin with a
        # different family or C++ standard library than the host changes the
        # toolchain tag and the host will reject the binary; inheriting the
        # default keeps the tag matched on every platform.
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

            # HVD engine dependency: single-precision FFTW (fftw3f).
            # nixpkgs exposes single precision as fftwFloat; using plain
            # `fftw` only provides the default/double-precision variant.
            fftwFloat

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
            # Probe via $CXX, not a hardcoded g++: the Darwin stdenv is
            # clang-based and has no g++, so the name would fall through to
            # Apple's /usr/bin/g++ xcrun shim and fail.
            echo "C++ toolchain: $($CXX --version | head -n1)"

            # Set up ccache if available
            export CMAKE_CXX_COMPILER_LAUNCHER=ccache

            # Make the single-precision FFTW prefix explicit for CMake.
            # This also makes the fallback find_library/find_path logic
            # deterministic on CI instead of depending on host search paths.
            export FFTW_DIR="${pkgs.fftwFloat}"
            export FFTW3F_ROOT="${pkgs.fftwFloat}"
          '';

          CMAKE_EXPORT_COMPILE_COMMANDS = 1;
          # Default to Ninja when no -G is given, matching decode-orc.
          CMAKE_GENERATOR = "Ninja";
        };
      }
    );
}
