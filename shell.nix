{
  pkgs ?
    import
    (builtins.fetchTarball {
      url = "https://github.com/nixos/nixpkgs/archive/ff4d5bb13c4ff6e3fbe4604ebcb7eb4c61bbee1f.tar.gz";
      sha256 = "1jfbdhqpi2r1akac39q0cmvaah8v9bfi4m1sxnfzkq5jamhldnvh";
    })
    {
      config = {
        allowUnfree = true;
      };
    },
}: let
  py_pkgs = ps:
    with ps; [
      # dev tools
      pip
      debugpy
      black
      isort

      # project specific
      numpy
      pandas
      scikit-learn
      torch-bin
      tqdm
    ];
  py = pkgs.python3.withPackages py_pkgs;
  debug = pkgs.writeShellScriptBin "debugpy-adapter" ''
    #!/usr/bin/env bash
    exec "python" -m debugpy.adapter "$@"
  '';
in
  pkgs.mkShell {
    packages = [
      py
      debug
    ];
  }
