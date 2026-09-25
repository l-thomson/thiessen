# Recipes that take enough arguments to be worth a name. Every other gate
# is a plain cargo command and is documented as one in CONTRIBUTING.md.

# Same-machine A/B of the wall-clock benchmarks over two revisions.
perf-compare rev-a rev-b filter="":
    tools/perf-compare.sh {{rev-a}} {{rev-b}} {{filter}}

# The wall-clock benchmarks on the working tree, saved under `working`.
perf-bench filter="":
    cargo bench --locked --manifest-path bench/Cargo.toml \
        --bench wall_clock -- --save-baseline working {{filter}}

# The instruction counts. Needs valgrind and gungraun-runner; the counts
# are deterministic, so this is the measurement a gate can read.
perf-instructions:
    cargo bench --locked --manifest-path bench/Cargo.toml \
        --bench instructions
