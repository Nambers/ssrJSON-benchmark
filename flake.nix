{
  description = "A simple flake for a simple python environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    ssrjson_ = {
      url = "github:antares0982/ssrjson";
      inputs.nixpkgs.follows = "nixpkgs";
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
          ssrjson = ssrjson_.packages.${pkgs.stdenv.hostPlatform.system}.ssrjson-pypackage-py313;
        in
        {
          default = pkgs.callPackage ./dev_tools/nix_files/shell.nix {
            persist = true;
            inherit ssrjson;
          };
        }
      );
      packages = forAllSystems (
        pkgs:
        let
          ssrjson = ssrjson_.packages.${pkgs.stdenv.hostPlatform.system}.ssrjson-pypackage-py313;
        in
        rec {
          default = pkgs.callPackage ./dev_tools/nix_files/package.nix { inherit ssrjson; };
          inherit ssrjson;
        }
      );
    };
}
