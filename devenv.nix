{ pkgs, lib, config, inputs, ... }:
{
  packages = with pkgs; [ 
    git
    gh
    pkgs.portmidi
    pkgs.pkg-config
    pkgs.libpng
    pkgs.libjpeg
    pkgs.freetype
    pkgs.fontconfig
    pkgs.SDL2_ttf
    pkgs.SDL2_mixer
    pkgs.SDL2_image
    pkgs.SDL2
   ];
   languages.python.enable = true;
   languages.python.poetry.enable = true;
}
