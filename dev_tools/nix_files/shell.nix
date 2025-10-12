{
  pkgs ? import <nixpkgs> { },
  lib ? pkgs.lib,
  persist ? false,
  mkShell ? pkgs.mkShell,
  callPackage ? pkgs.callPackage,
  ssrjson,
  ...
}:
let
  optionalAttrs = lib.attrsets.optionalAttrs;
  # define the nix-pyenv directory
  nix-pyenv-directory = ".nix-pyenv";
  # define version
  usingPython = pkgs.python313;
  # import required python packages
  requiredPythonPackages = callPackage ./py_requirements.nix { inherit ssrjson; };
  # create python environment
  pyenv = usingPython.withPackages requiredPythonPackages;
  #
  callShellHookParam = {
    inherit
      nix-pyenv-directory
      pyenv
      usingPython
      persist
      pkgs
      ;
  };
  internalShell = mkShell (
    {
      packages = [
        pyenv
        pkgs.cmake
      ];
    }
    // (optionalAttrs (!persist) {
      shellHook = callPackage ./shellhook.nix callShellHookParam;
    })
  );
in
internalShell.overrideAttrs (
  optionalAttrs persist {
    shellHook = callPackage ./shellhook.nix (
      callShellHookParam
      // {
        inherit (internalShell) inputDerivation;
      }
    );
  }
)
