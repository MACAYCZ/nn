{
  pkgs ? import <nixpkgs> {
    config.allowUnfree = true;
    config.cudaSupport = true;
  },
}:
pkgs.mkShell.override { stdenv = pkgs.gcc12Stdenv; } {
  packages = with pkgs; [
    cmake
    cudaPackages.cudatoolkit
  ];
  LD_LIBRARY_PATH = "/run/opengl-driver/lib";
}
