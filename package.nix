{
  pkgs,
  python3Packages,
  cmake,
  ssrjson,
  ...
}:
let
  requirements = (pkgs.callPackage ./py_requirements.nix { inherit ssrjson; }) python3Packages;
in
python3Packages.buildPythonPackage {
  pname = "ssrjson-benchmark";
  src = builtins.path {
    path = ./.;
    name = "ssrjson-benchmark-src";
  };
  version = builtins.readFile ./version_file;
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
