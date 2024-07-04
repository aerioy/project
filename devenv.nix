{ pkgs, lib, config, inputs, ... }:

{
  # https://devenv.sh/packages/
  packages = with pkgs; [ git gh
  
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

  languages.javascript.enable = true;
   languages.python.enable = true;
   languages.python.poetry.enable = true;


  services.postgres = {
    enable = true;
    listen_addresses = "0.0.0.0";
    initialScript = "CREATE ROLE postgres SUPERUSER; ALTER ROLE postgres WITH LOGIN;";
    initialDatabases = [{ name = "dbthing"; }];
  };
 
  enterShell = ''
  '';

  # https://devenv.sh/tests/
  enterTest = ''
  '';
}
