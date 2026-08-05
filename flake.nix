# A nix package and dev shell.
#
#     nix run . -- info <model>
#     nix build            # result/bin/moe
#     nix develop          # the toolchain the CI uses, plus vhs for the demos
#
# The engine has no system dependencies beyond a C compiler for the TLS stack, so
# this is almost the default rustPlatform build; the interesting part is that
# `cargo test` needs no network, which lets the checkPhase run inside the sandbox.
{
  description = "CPU inference for sparse mixture-of-experts models";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAll = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      packages = forAll (pkgs: rec {
        moe = pkgs.rustPlatform.buildRustPackage {
          pname = "moe";
          version = (builtins.fromTOML (builtins.readFile ./Cargo.toml)).package.version;
          src = ./.;
          cargoLock.lockFile = ./Cargo.lock;
          nativeBuildInputs = [ pkgs.pkg-config ];
          # The suite downloads nothing, so it runs in the sandbox as-is.
          doCheck = true;
          meta = {
            description = "CPU inference for sparse mixture-of-experts language models";
            homepage = "https://github.com/JGalego/Moe";
            license = nixpkgs.lib.licenses.mit;
            mainProgram = "moe";
          };
        };
        default = moe;
      });

      devShells = forAll (pkgs: {
        default = pkgs.mkShell {
          packages = with pkgs; [
            rustc
            cargo
            rustfmt
            clippy
            pkg-config
            # For scripts/oracle.py and scripts/tokcheck.py.
            python3
            # For recording the tapes in tapes/.
            vhs
          ];
        };
      });

      apps = forAll (pkgs: {
        default = {
          type = "app";
          program = "${self.packages.${pkgs.system}.moe}/bin/moe";
        };
      });
    };
}
