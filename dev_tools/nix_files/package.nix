{
  pkgs,
  python3Packages,
  cmake,
  ssrjson,
  ...
}:
let
  requirements = (pkgs.callPackage ./py_requirements.nix { inherit ssrjson; }) python3Packages;
  findVersion = pkgs.callPackage ./find_version.nix { };
  version = findVersion ./../../pyproject.toml;
in
python3Packages.buildPythonPackage {
  pname = "ssrjson-benchmark";
  src = builtins.path {
    path = ./../..;
    name = "ssrjson-benchmark-src";
  };
  inherit version;
  pyproject = true;

  nativeBuildInputs = [
    cmake
    python3Packages.setuptools
  ];

  dependencies = requirements;

  preBuild = ''
    cd ..
  '';
}
