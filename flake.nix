{
  description = "python venv for my packages";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { system = system; config.allowUnfree = true; config.cudaSupport = true; };
      venvDir = "venv";
    in {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          virtualenv
          python312
          python312Packages.jupyterlab
          python312Packages.pandas
          python312Packages.numpy
          python312Packages.scipy
          python312Packages.networkx
          python312Packages.matplotlib
          python312Packages.plotly
        ];
      };
  };
}