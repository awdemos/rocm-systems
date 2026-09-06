# rocm-systems

## OVERVIEW

Super-repo consolidating multiple ROCm systems projects (ROCm runtime, profilers, debug agents, tests, etc.) into a single repository for unified builds, CI, and integration. Default branch is `develop`.

## STRUCTURE

```
projects/           Individual ROCm systems projects (clr, hip, rocprofiler*, rocr-runtime, rocm-core, rocminfo, ...)
shared/             Dependencies used by multiple projects that do not ship distinct packages
emulation/          ROCm emulation projects (mirage, rocjitsu)
experimental/       Experimental projects (perf-dkms)
profilers/          Profiler-hub integration tools
tools/              Shared tooling
python/             Python helpers for the super-repo
docs/               Contributor docs (migration, CI, gardening, PR bot FAQ)
.github/            GitHub Actions workflows and PR helper scripts
```

## COMMANDS

```bash
python3 -c "import yaml; yaml.safe_load(open('.pre-commit-config.yaml'))"   # Verify root pre-commit config is valid YAML
cat .github/requirements.txt                                                  # Python deps for GitHub helper scripts
```

Project-specific commands vary by subdirectory; most use CMake or Python. Example:

```bash
cd projects/rocminfo
mkdir build && cd build
cmake ..
make
```

## SETUP

- This is a multi-language super-repo (C/C++, CMake, Python, Go in some subprojects).
- Many `projects/` directories are Git submodules or subtree imports; see `.gitmodules`.
- Read `CONTRIBUTING.md` for sparse-checkout setup and the monorepo workflow.
- Install `pre-commit` and run `pre-commit run --all-files` at the root for YAML/trailing-whitespace checks.
- Project-specific dependencies are documented in each `projects/<name>/README.md` or `CONTRIBUTING.md`.

## CODE STYLE

- Each subproject follows its own conventions; check the project's `CONTRIBUTING.md` and `pyproject.toml`/`CMakeLists.txt`.
- Root-level Python helper scripts should pass `pre-commit` hooks (trailing whitespace, EOF, YAML checks).

## DEPLOYMENT

GitHub Actions drive CI. The `develop` branch is the integration branch; component workflows live under `.github/workflows/`. TheRock CI provides multi-component builds on top of the TheRock build system.
