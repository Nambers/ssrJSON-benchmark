{
  pkgs,
  python3Packages,
  cmake,
  ssrjson,
  ...
}:
let
  requirements = (pkgs.callPackage ./py_requirements.nix { }) python3Packages;
in
python3Packages.buildPythonPackage {
  pname = "ssrjson-benchmark";
  src = builtins.path {
    path = ./.;
    name = "ssrjson-benchmark-src";
  };
  version = "0.0.3";
  pyproject = true;

  nativeBuildInputs = requirements ++ [
    cmake
    python3Packages.setuptools
  ];

  preBuild = ''
    cd ..
  '';
  pythonRuntimeDepsCheckHook = ''
    export PYTHONPATH=$PYTHONPATH:${ssrjson}
  '';
}
