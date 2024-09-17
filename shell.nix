let
  nixpkgs = fetchTarball "https://github.com/NixOS/nixpkgs/tarball/nixos-24.05";
  pkgs = import nixpkgs {
    config.allowUnfree = true;
    config.cudaSupport = true;
  };
in
  pkgs.mkShell.override {
    stdenv = pkgs.gcc12Stdenv;
  } {
    packages = with pkgs; [
      cmake
      cudaPackages.cudatoolkit
      raylib
    ];
    LD_LIBRARY_PATH = "/run/opengl-driver/lib";
  }
