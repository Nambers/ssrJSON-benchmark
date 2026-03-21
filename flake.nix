{
  description = "A simple flake for a simple python environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    ssrjson-nix-dev = {
      url = "github:antares0982/ssrjson-nix-dev";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    ssrjson_ = {
      url = "github:antares0982/ssrjson";
      inputs.ssrjson-nix-dev.follows = "ssrjson-nix-dev";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      ssrjson_,
      ...
    }:
    let
      py-minor-ver = 14;
      py-minor-ver-str = builtins.toString py-minor-ver;
      forAllSystems =
        function:
        nixpkgs.lib.genAttrs
          [
            "x86_64-linux"
            "aarch64-linux"
            "aarch64-darwin"
          ]
          (
            system:
            function (
              import nixpkgs {
                inherit system;
              }
            )
          );
    in
    {
      devShells = forAllSystems (
        pkgs:
        let
          ssrjson = ssrjson_.packages.${pkgs.stdenv.hostPlatform.system}.ssrjson-pypackage-py314;
        in
        {
          default = pkgs.callPackage ./dev_tools/nix_files/shell.nix {
            persist = true;
            inherit ssrjson py-minor-ver-str;
          };
        }
      );
      packages = forAllSystems (
        pkgs:
        let
          ssrjson =
            ssrjson_.packages.${pkgs.stdenv.hostPlatform.system}."ssrjson-pypackage-py3${py-minor-ver-str}";
        in
        rec {
          default = pkgs.callPackage ./dev_tools/nix_files/package.nix {
            python = pkgs.python314;
          };
          inherit ssrjson;
        }
      );
    };
}
