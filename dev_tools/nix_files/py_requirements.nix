{ pkgs, ssrjson, ... }:
pypkgs: with pypkgs; [
  matplotlib
  orjson
  psutil
  reportlab
  svglib
  build
  pip
  ujson
  msgspec
  ssrjson
]
