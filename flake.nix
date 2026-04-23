# flake.nix
{
  description = "Hack the World – Adversarial drone-defense simulation engine & visualizer";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        # ── Python layer ────────────────────────────────────────────────
        # We override the Python package set at the root level.
        # Now, ANY package that requests these dependencies will get our modified versions.
        python = pkgs.python312.override {
          packageOverrides = final: prev: {
            jax = prev.jax.overridePythonAttrs (old: {
              doCheck = false;
              nativeCheckInputs = [];
            });
            
            jaxlib = prev.jaxlib.overridePythonAttrs (old: {
              doCheck = false;
              nativeCheckInputs = [];
            });

            gymnasium = (prev.gymnasium.override {
              mujoco = null;
            }).overridePythonAttrs (old: {
              doCheck = false;
            });

            pettingzoo = prev.pettingzoo.overridePythonAttrs (old: {
              doCheck = false;
            });
          };
        };

        # Because we handled the overrides above, this list becomes purely declarative.
        pythonEnv = python.withPackages (ps: with ps; [
          # Core simulation
          numpy
          gymnasium
          pettingzoo
          jax
          jaxlib

          # City / graph topology
          networkx
          shapely

          # Web bridge (FastAPI + WebSockets)
          fastapi
          uvicorn
          websockets

          # Experiment bookkeeping
          orjson
          httpx
          pydantic
          pyparsing

          # Visualization
          matplotlib
          pillow

          # Dev / QoL
          pytest
          black
          ruff
          ipython
          jupyter
        ]);

        # ── Node layer (visualizer) ────────────────────────────────────
        node = pkgs.nodejs_20;

      in
      {
        devShells.default = pkgs.mkShell {
          name = "hack-the-world";

          buildInputs = [
            pythonEnv
            node
            pkgs.pnpm
            pkgs.just
            pkgs.jq
            pkgs.ocl-icd
            pkgs.level-zero
            pkgs.intel-compute-runtime
            pkgs.intel-compute-runtime.drivers
          ];

          shellHook = ''
            echo ""
            echo "  ╔══════════════════════════════════════════════╗"
            echo "  ║  Hack the World – dev shell ready            ║"
            echo "  ║  Python : $(python --version 2>&1)                  ║"
            echo "  ║  Node   : $(node --version)                     ║"
            echo "  ╚══════════════════════════════════════════════╝"
            echo ""

            export PYTHONDONTWRITEBYTECODE=1
            export INTEL_COMPUTE_RUNTIME_ROOT=${pkgs.intel-compute-runtime}
            export INTEL_COMPUTE_RUNTIME_DRIVERS=${pkgs.intel-compute-runtime.drivers}
            export LEVEL_ZERO_ROOT=${pkgs.level-zero}
            export OCL_ICD_VENDORS="$INTEL_COMPUTE_RUNTIME_ROOT/etc/OpenCL/vendors"
            export INTEL_GCC_RUNTIME="$(nix-store -q --references "$INTEL_COMPUTE_RUNTIME_ROOT" | grep 'gcc-.*-lib' | head -n1 || true)"
            EXTRA_LD_LIBRARY_PATH="$INTEL_COMPUTE_RUNTIME_ROOT/lib:$INTEL_COMPUTE_RUNTIME_ROOT/lib/intel-opencl:$INTEL_COMPUTE_RUNTIME_DRIVERS/lib:$LEVEL_ZERO_ROOT/lib"
            if [ -n "$INTEL_GCC_RUNTIME" ]; then
              EXTRA_LD_LIBRARY_PATH="$EXTRA_LD_LIBRARY_PATH:$INTEL_GCC_RUNTIME/lib"
            fi
            if [ -n "$LD_LIBRARY_PATH" ]; then
              EXTRA_LD_LIBRARY_PATH="$EXTRA_LD_LIBRARY_PATH:$LD_LIBRARY_PATH"
            fi
            export LD_LIBRARY_PATH="$EXTRA_LD_LIBRARY_PATH"

            if [ -f .tmp/intel-jax.env ]; then
              source .tmp/intel-jax.env
            fi

            [ -f .env ] && set -a && source .env && set +a
          '';
        };

        checks.default = pkgs.runCommand "pytest" {
          buildInputs = [ pythonEnv ];
        } ''
          cd ${self}
          python -m pytest tests/ -q --tb=short
          touch $out
        '';
      }
    );
}
