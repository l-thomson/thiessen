#!/bin/sh
# Vendors the Rust sources the R package builds from, offline, as CRAN
# requires: the core crate as `cargo package` publishes it, less its
# `[dev-dependencies]` table, and every third-party crate through
# `cargo vendor`. Writes r/src/rust/core as source, r/src/rust/vendor.tar.xz
# holding the third-party crates alone, r/inst/AUTHORS, and the core version
# in r/DESCRIPTION.
#
# Usage, from the repository root: tools/vendor.sh
set -eu

root=$(cd "$(dirname "$0")/.." && pwd)
rust="$root/r/src/rust"
cd "$root"

version=$(cargo metadata --no-deps --format-version 1 \
  | jq -r '.packages[] | select(.name == "thiessen") | .version')
cargo package -p thiessen --no-verify --allow-dirty --quiet
rm -rf "$rust/core" "$rust/vendor"
mkdir -p "$rust/core"
tar xzmf "target/package/thiessen-$version.crate" --strip-components=1 -C "$rust/core"
# The test-only crates are not built by the R package and would otherwise
# be vendored; `cargo vendor` has no switch for dev-dependencies.
awk '/^\[dev-dependencies/ { skip = 1; next } /^\[/ { skip = 0 } !skip' \
  "$rust/core/Cargo.toml" > "$rust/core/Cargo.toml.tmp"
mv "$rust/core/Cargo.toml.tmp" "$rust/core/Cargo.toml"
rm -f "$rust/core/Cargo.toml.orig" "$rust/core/.cargo_vcs_info.json" "$rust/core/Cargo.lock"

cd "$rust"
if [ -f Cargo.lock ]; then locked="--locked"; else locked=""; fi
cargo vendor $locked --quiet vendor >/dev/null

# The XAddiVortes cross-binding fixture: the R fit must equal the core's.
cp "$root/crates/thiessen/tests/fixtures/xaddivortes/cross_binding.txt" \
  "$root/r/tests/testthat/core-xaddivortes-fixture.txt"

cargo metadata --format-version 1 --locked | jq -r '
  "Third-party Rust crates vendored under src/rust/vendor, with the",
  "authors and licences their manifests declare.",
  "",
  (.packages
   | map(select(.name != "thiessen" and .name != "thiessen-r"))
   | sort_by(.name, .version)[]
   | "\(.name) \(.version)",
     "  authors: \(if (.authors | length) > 0 then (.authors | join(", ")) else "not declared" end)",
     "  license: \(.license)",
     (if .repository then "  repository: \(.repository)" else empty end),
     "")' > "$root/r/inst/AUTHORS"

rm -f vendor.tar.xz
tar --sort=name --owner=0 --group=0 --numeric-owner --mtime='2000-01-01 00:00Z' \
  -cJf vendor.tar.xz vendor

# The determinism test compares against the core's committed chain, which
# the package tarball does not otherwise carry.
cp "$root/crates/thiessen/tests/chains/gaussian.txt" \
  "$root/r/tests/testthat/core-gaussian-chain.txt"
cp "$root/crates/thiessen/tests/chains/probit.txt" \
  "$root/r/tests/testthat/core-probit-chain.txt"
cp "$root/crates/thiessen/tests/chains/heteroscedastic.txt" \
  "$root/r/tests/testthat/core-heteroscedastic-chain.txt"

sed -i "s|^Config/thiessen/core-version: .*|Config/thiessen/core-version: $version|" \
  "$root/r/DESCRIPTION"

printf 'core %s; vendor.tar.xz %s\n' "$version" "$(du -h vendor.tar.xz | cut -f1)"
