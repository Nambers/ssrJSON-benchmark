{ pkgs, ... }:
pypkgs: with pypkgs; [
  matplotlib
  orjson
  psutil
  reportlab
  svglib
  build
  pip
]
