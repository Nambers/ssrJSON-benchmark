{
  pkgs,
  python,
  cmake,
  callPackage,
  ...
}:
let
  findVersion = callPackage ./find_version.nix { };
  version = findVersion ./../../pyproject.toml;
  pypkgs = python.pkgs;
in
pypkgs.buildPythonPackage {
  pname = "ssrjson-benchmark";
  src = builtins.path {
    path = ./../..;
    name = "ssrjson-benchmark-src";
  };
  inherit version;
  pyproject = true;

  nativeBuildInputs = [
    cmake
    pypkgs.setuptools
  ];

  preBuild = ''
    cd ..
  '';
}
