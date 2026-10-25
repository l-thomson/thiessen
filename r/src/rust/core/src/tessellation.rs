//! A Voronoi tessellation of a covariate subspace and the assignment of
//! observations to its cells.

use crate::data::Data;
use crate::error::{Error, Result};
use crate::geometry::Geometry;

/// One tessellation: b centres in a d-dimensional subspace of the scaled
/// covariate space, one cell mean per centre.
#[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "TessellationParts")]
pub struct Tessellation {
    /// Row-major b by d centre coordinates (scaled space).
    pub(crate) centres: Vec<f64>,
    /// The d active covariates, zero-based column indices, in centre
    /// coordinate order.
    pub(crate) dims: Vec<usize>,
    /// Cell means mu_1..mu_b (scaled space); under the linear cell basis
    /// the intercepts, the value at the cell's centre.
    pub(crate) mus: Vec<f64>,
    /// Row-major b by d slopes of the linear cell basis (scaled space);
    /// empty under the constant basis.
    pub(crate) betas: Vec<f64>,
    /// The soft-membership kernel bandwidth (scaled space); `None` under
    /// hard membership. Experimental (`docs/experimental.md`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) tau: Option<f64>,
}

impl Clone for Tessellation {
    fn clone(&self) -> Self {
        Self {
            centres: self.centres.clone(),
            dims: self.dims.clone(),
            mus: self.mus.clone(),
            betas: self.betas.clone(),
            tau: self.tau,
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.centres.clone_from(&source.centres);
        self.dims.clone_from(&source.dims);
        self.mus.clone_from(&source.mus);
        self.betas.clone_from(&source.betas);
        self.tau = source.tau;
    }
}

impl Tessellation {
    /// No cells and no dimensions: a buffer a proposal is written into.
    pub(crate) fn empty() -> Self {
        Self {
            centres: Vec::new(),
            dims: Vec::new(),
            mus: Vec::new(),
            betas: Vec::new(),
            tau: None,
        }
    }

    /// Number of cells b.
    pub fn n_cells(&self) -> usize {
        self.mus.len()
    }

    /// Number of active covariates d.
    pub fn n_dims(&self) -> usize {
        self.dims.len()
    }

    /// The active covariates, zero-based column indices.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Cell means, scaled space.
    pub fn mus(&self) -> &[f64] {
        &self.mus
    }

    /// Centre `k`'s coordinates, one per active covariate.
    pub fn centre(&self, k: usize) -> &[f64] {
        let d = self.dims.len();
        &self.centres[k * d..(k + 1) * d]
    }

    /// Squared distance under `geometry` from `row` (a full p-length scaled
    /// row) to centre `k` over the active covariates.
    pub(crate) fn key(&self, row: &[f64], k: usize, geometry: &Geometry) -> f64 {
        geometry.key(row, &self.dims, self.centre(k))
    }

    /// The nearest centre to `row` and its key; ties go to the lowest index.
    pub(crate) fn nearest(&self, row: &[f64], geometry: &Geometry) -> (usize, f64) {
        let mut best = f64::INFINITY;
        let mut best_cell = 0;
        for k in 0..self.n_cells() {
            let key = self.key(row, k, geometry);
            if key < best {
                best = key;
                best_cell = k;
            }
        }
        (best_cell, best)
    }

    /// The soft-membership kernel bandwidth; `None` under hard
    /// membership, and in a build without the `experimental` feature.
    pub fn bandwidth(&self) -> Option<f64> {
        self.tau
    }

    /// The value of the tessellation at `row`: the mean of the cell `row`
    /// falls in, tilted by the cell's slopes under the linear basis;
    /// under soft membership the kernel-weighted sum of the cell means.
    pub(crate) fn value_at(&self, row: &[f64], geometry: &Geometry) -> f64 {
        if let Some(tau) = self.tau {
            return self.soft_value_at(row, geometry, tau);
        }
        self.value_in_cell(self.nearest(row, geometry).0, row)
    }

    /// The value at every row of `x` in row order, passed to `f` with the
    /// row index. Under hard membership on an all-Euclidean geometry the
    /// cells come from one streaming assignment over the column-major
    /// design, with `keys` as its scratch, grown once and reused across
    /// calls; otherwise each row is evaluated as [`value_at`](Self::value_at).
    pub(crate) fn for_each_value(
        &self,
        x: &Data,
        geometry: &Geometry,
        keys: &mut Vec<f64>,
        mut f: impl FnMut(usize, f64),
    ) {
        let n = x.n_rows();
        if n == 0 {
            return;
        }
        if self.tau.is_none() && geometry.is_plain() {
            keys.clear();
            nearest_blocks(x, self, |start, cells| {
                for (i, &cell) in (start..).zip(cells) {
                    f(i, self.value_in_cell(cell as usize, x.row(i)));
                }
            });
        } else {
            for i in 0..n {
                f(i, self.value_at(x.row(i), geometry));
            }
        }
    }

    /// The kernel-weighted sum of the cell means at `row`: weights
    /// proportional to exp(-key / (2 tau^2)) over the centres, computed
    /// from the smallest key so the nearest centre's factor is 1.
    fn soft_value_at(&self, row: &[f64], geometry: &Geometry, tau: f64) -> f64 {
        let b = self.n_cells();
        let mut keys = Vec::with_capacity(b);
        let mut min = f64::INFINITY;
        for k in 0..b {
            let key = self.key(row, k, geometry);
            min = min.min(key);
            keys.push(key);
        }
        let scale = 1.0 / (2.0 * tau * tau);
        let mut total = 0.0;
        let mut value = 0.0;
        for (key, &mu) in keys.iter().zip(&self.mus) {
            let g = crate::maths::exp(-(key - min) * scale);
            total += g;
            value += g * mu;
        }
        value / total
    }

    /// The value cell `cell` contributes at `row`: the cell mean, plus,
    /// under the linear basis, the slopes against the row's offset from
    /// the cell's centre on the active covariates.
    #[inline]
    pub(crate) fn value_in_cell(&self, cell: usize, row: &[f64]) -> f64 {
        let mut value = self.mus[cell];
        if !self.betas.is_empty() {
            let d = self.dims.len();
            let centre = self.centre(cell);
            for (j, &dim) in self.dims.iter().enumerate() {
                value += self.betas[cell * d + j] * (row[dim] - centre[j]);
            }
        }
        value
    }

    /// [`value_in_cell`](Self::value_in_cell) at row `i` of `x`, read
    /// only under the linear basis.
    pub(crate) fn training_value(&self, cell: usize, x: &Data, i: usize) -> f64 {
        if self.betas.is_empty() {
            self.mus[cell]
        } else {
            self.value_in_cell(cell, x.row(i))
        }
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TessellationParts {
    centres: Vec<f64>,
    dims: Vec<usize>,
    mus: Vec<f64>,
    #[serde(default)]
    betas: Vec<f64>,
    #[serde(default)]
    tau: Option<f64>,
}

impl TryFrom<TessellationParts> for Tessellation {
    type Error = Error;

    fn try_from(parts: TessellationParts) -> Result<Self> {
        let bad = |reason: &str| Error::InvalidSavedModel {
            reason: reason.into(),
        };
        let (b, d) = (parts.mus.len(), parts.dims.len());
        if b == 0 || d == 0 {
            return Err(bad(
                "a tessellation needs at least one cell and one dimension",
            ));
        }
        if parts.centres.len() != b * d {
            return Err(bad(
                "centre buffer length does not match cells by dimensions",
            ));
        }
        if !parts.betas.is_empty() && parts.betas.len() != b * d {
            return Err(bad(
                "slope buffer length does not match cells by dimensions",
            ));
        }
        if parts
            .centres
            .iter()
            .chain(&parts.mus)
            .chain(&parts.betas)
            .any(|v| !v.is_finite())
        {
            return Err(bad("tessellation values must be finite"));
        }
        for (i, dim) in parts.dims.iter().enumerate() {
            if parts.dims[..i].contains(dim) {
                return Err(bad("tessellation dimensions must be distinct"));
            }
        }
        if parts.tau.is_some_and(|tau| !(tau.is_finite() && tau > 0.0)) {
            return Err(bad(
                "the soft-membership bandwidth must be finite and positive",
            ));
        }
        Ok(Self {
            centres: parts.centres,
            dims: parts.dims,
            mus: parts.mus,
            betas: parts.betas,
            tau: parts.tau,
        })
    }
}

/// The structural change a proposal makes relative to the tessellation it
/// was proposed from, so the cached assignment can be updated rather than
/// recomputed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Delta {
    /// One centre appended at the highest index; dims unchanged.
    CentreAdded,
    /// One centre removed; higher indices shift down by one; dims unchanged.
    CentreRemoved(usize),
    /// One centre's coordinates changed in place; dims unchanged.
    CentreMoved(usize),
    /// dims changed: every key is stale.
    Full,
}

/// A row's cell index in an [`Assignment`]: 4 bytes per row and
/// tessellation. An index is below its tessellation's centre count, which
/// is below 2^32: the coordinates of 2^32 centres alone would take 32 GiB.
pub(crate) type CellIndex = u32;

/// Cached nearest-centre assignment of every observation, with the winning
/// key, for one tessellation; under soft membership also every key.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct Assignment {
    /// Cell index per observation.
    pub(crate) cells: Vec<CellIndex>,
    /// Squared distance to the assigned centre per observation; empty
    /// under hard membership on an all-Euclidean geometry, whose moves
    /// recompute it column by column in the arithmetic that formed it.
    keys: Vec<f64>,
    /// Column-major b by n keys of every observation against every
    /// centre; `None` under hard membership, whose moves need only the
    /// winner.
    soft: Option<Vec<f64>>,
}

/// Rows per block of [`nearest_blocks`].
const BLOCK: usize = 256;

/// Rows whose keys and winners stay in registers while every centre is
/// compared.
const LANES: usize = 8;

/// The nearest centre of every row of `x` under `t`, an all-Euclidean
/// hard-membership tessellation, `BLOCK` rows at a time: `emit(start,
/// cells)` receives each block. Each key sums the squared differences in
/// `dims` order, the sum [`Geometry::key`] forms for an all-Euclidean
/// geometry, and the centres are compared in index order with the strict
/// comparison of [`nearest_plain`], so ties go to the lowest index. `x`
/// must have at least one row.
pub(crate) fn nearest_blocks(
    x: &Data,
    t: &Tessellation,
    mut emit: impl FnMut(usize, &[CellIndex]),
) {
    let n = x.n_rows();
    let mut cells = [0 as CellIndex; BLOCK];
    let mut start = 0;
    while start < n {
        let cells = &mut cells[..BLOCK.min(n - start)];
        nearest_in_block(x, start, t, cells);
        emit(start, cells);
        start += cells.len();
    }
}

/// The nearest centre of `t` to rows `start..start + cells.len()` of `x`,
/// into `cells`: `LANES` rows at a time, then [`nearest_plain`] row by
/// row.
fn nearest_in_block(x: &Data, start: usize, t: &Tessellation, cells: &mut [CellIndex]) {
    let n = x.n_rows();
    let columns = x.columns();
    let full = cells.len() - cells.len() % LANES;
    let column = |j: usize| &columns[t.dims[j] * n + start..][..full];
    let (lanes, rest) = cells.split_at_mut(full);
    match t.dims.len() {
        1 => nearest_in_columns([column(0)], &t.centres, lanes),
        2 => nearest_in_columns([column(0), column(1)], &t.centres, lanes),
        3 => nearest_in_columns([column(0), column(1), column(2)], &t.centres, lanes),
        _ => {
            for (i, out) in (start..).step_by(LANES).zip(lanes.chunks_exact_mut(LANES)) {
                let winners = nearest_in_lanes_any(columns, n, i, t);
                for (out, cell) in out.iter_mut().zip(winners) {
                    *out = cell as CellIndex;
                }
            }
        }
    }
    for (i, out) in (start + full..).zip(rest) {
        *out = nearest_plain(t, x.row(i)).0 as CellIndex;
    }
}

/// [`nearest_in_block`] on `D` active columns of a whole number of
/// `LANES` rows.
#[inline(always)]
fn nearest_in_columns<const D: usize>(x: [&[f64]; D], centres: &[f64], cells: &mut [CellIndex]) {
    for (i, out) in (0..).step_by(LANES).zip(cells.chunks_exact_mut(LANES)) {
        let rows: [&[f64; LANES]; D] =
            std::array::from_fn(|j| x[j][i..i + LANES].try_into().expect("LANES rows"));
        let (first, rest) = centres.split_at(D);
        let mut nearest = Nearest::new(lane_keys(rows, first));
        for centre in rest.chunks_exact(D) {
            nearest.offer(lane_keys(rows, centre));
        }
        for (out, cell) in out.iter_mut().zip(nearest.cells) {
            *out = cell as CellIndex;
        }
    }
}

/// The keys of `LANES` rows, one array per active column, against
/// `centre`.
#[inline(always)]
fn lane_keys<const D: usize>(x: [&[f64; LANES]; D], centre: &[f64]) -> [f64; LANES] {
    let mut key = [0.0; LANES];
    for (key, &x) in key.iter_mut().zip(x[0]) {
        let diff = x - centre[0];
        *key = diff * diff;
    }
    for (x, &c) in x.iter().zip(centre).skip(1) {
        for (key, &x) in key.iter_mut().zip(*x) {
            let diff = x - c;
            *key += diff * diff;
        }
    }
    key
}

/// The nearest centre of `t` to rows `i..i + LANES` of the column-major
/// `columns` of `n` rows, on any number of active columns.
#[inline(always)]
fn nearest_in_lanes_any(columns: &[f64], n: usize, i: usize, t: &Tessellation) -> [u64; LANES] {
    let mut centres = t.centres.chunks_exact(t.dims.len());
    let first = centres.next().expect("a centre");
    let mut nearest = Nearest::new(lane_keys_any(columns, n, i, &t.dims, first));
    for centre in centres {
        nearest.offer(lane_keys_any(columns, n, i, &t.dims, centre));
    }
    nearest.cells
}

/// The keys of rows `i..i + LANES` of the column-major `columns` of `n`
/// rows against `centre` on `dims`. Summing from zero leaves each key as
/// [`lane_keys`] forms it, since every term is at least +0.
#[inline(always)]
fn lane_keys_any(
    columns: &[f64],
    n: usize,
    i: usize,
    dims: &[usize],
    centre: &[f64],
) -> [f64; LANES] {
    let mut key = [0.0; LANES];
    for (&dim, &c) in dims.iter().zip(centre) {
        let x: &[f64; LANES] = columns[dim * n + i..][..LANES]
            .try_into()
            .expect("LANES rows");
        for (key, &x) in key.iter_mut().zip(x) {
            let diff = x - c;
            *key += diff * diff;
        }
    }
    key
}

/// The running nearest centre of `LANES` rows over centres offered in
/// index order. The cells are held in lanes as wide as the keys so that
/// the comparison and the selection vectorise together.
struct Nearest {
    keys: [f64; LANES],
    cells: [u64; LANES],
    next: u64,
}

impl Nearest {
    /// Centre 0's keys.
    #[inline(always)]
    fn new(keys: [f64; LANES]) -> Self {
        Self {
            keys,
            cells: [0; LANES],
            next: 1,
        }
    }

    /// The next centre's keys, taken where strictly smaller.
    #[inline(always)]
    fn offer(&mut self, keys: [f64; LANES]) {
        let lanes = self.keys.iter_mut().zip(&mut self.cells).zip(keys);
        for ((best, cell), key) in lanes {
            let nearer = key < *best;
            *best = if nearer { key } else { *best };
            *cell = if nearer { self.next } else { *cell };
        }
        self.next += 1;
    }
}

/// [`Tessellation::nearest`] for an all-Euclidean geometry: the same sum
/// in the same order without the per-column metric dispatch.
fn nearest_plain(t: &Tessellation, row: &[f64]) -> (usize, f64) {
    let d = t.dims.len();
    let mut best = f64::INFINITY;
    let mut best_cell = 0;
    for (k, centre) in t.centres.chunks_exact(d).enumerate() {
        let mut key = 0.0;
        for (&dim, &c) in t.dims.iter().zip(centre) {
            let diff = row[dim] - c;
            key += diff * diff;
        }
        if key < best {
            best = key;
            best_cell = k;
        }
    }
    (best_cell, best)
}

/// The squared distance of every row of `x` to centre `k` of `t` into
/// `out`, one active column at a time in `dims` order over the
/// column-major design: the sum [`Geometry::key`] forms row by row for an
/// all-Euclidean geometry.
fn centre_keys(x: &Data, t: &Tessellation, k: usize, out: &mut [f64]) {
    let n = x.n_rows();
    let columns = x.columns();
    let out = &mut out[..n];
    for (j, (&dim, &c)) in t.dims.iter().zip(t.centre(k)).enumerate() {
        let column = &columns[dim * n..(dim + 1) * n];
        if j == 0 {
            for i in 0..n {
                let diff = column[i] - c;
                out[i] = diff * diff;
            }
        } else {
            for i in 0..n {
                let diff = column[i] - c;
                out[i] += diff * diff;
            }
        }
    }
}

/// The squared distance of every row of `x` to the centre `cells` assigns
/// it under `t`, into `out`, in the arithmetic of [`centre_keys`], so
/// equal to the key that chose the cell. `coordinate`, of one entry per
/// centre, holds one active column's centre coordinates at a time.
fn own_keys(
    x: &Data,
    t: &Tessellation,
    cells: &[CellIndex],
    coordinate: &mut [f64],
    out: &mut [f64],
) {
    let n = x.n_rows();
    let columns = x.columns();
    let d = t.dims.len();
    let (cells, out) = (&cells[..n], &mut out[..n]);
    for (j, &dim) in t.dims.iter().enumerate() {
        for (c, &v) in coordinate
            .iter_mut()
            .zip(t.centres.iter().skip(j).step_by(d))
        {
            *c = v;
        }
        let column = &columns[dim * n..(dim + 1) * n];
        if j == 0 {
            for i in 0..n {
                let diff = column[i] - coordinate[cells[i] as usize];
                out[i] = diff * diff;
            }
        } else {
            for i in 0..n {
                let diff = column[i] - coordinate[cells[i] as usize];
                out[i] += diff * diff;
            }
        }
    }
}

/// Replace each row's winner with centre `k` where `column` holds a
/// smaller key: the strict comparison of [`nearest_plain`], as selects
/// rather than a branch because the comparison is a coin flip per row.
/// The cell takes a mask rather than a select: on baseline x86-64 LLVM
/// turns a select of the index back into a branch and the loop stops
/// vectorising. Returns whether centre `k` took any row.
fn take_nearer(column: &[f64], k: usize, cells: &mut [CellIndex], best: &mut [f64]) -> bool {
    let n = cells.len();
    let (column, best) = (&column[..n], &mut best[..n]);
    let k = k as CellIndex;
    let mut taken = 0;
    for i in 0..n {
        let nearer = column[i] < best[i];
        let mask = (nearer as CellIndex).wrapping_neg();
        best[i] = if nearer { column[i] } else { best[i] };
        cells[i] ^= (cells[i] ^ k) & mask;
        taken |= mask;
    }
    taken != 0
}

/// `scratch` with at least `len` elements, its contents unspecified.
fn key_buffer(scratch: &mut Vec<f64>, len: usize) -> &mut [f64] {
    if scratch.len() < len {
        scratch.resize(len, 0.0);
    }
    &mut scratch[..len]
}

impl Clone for Assignment {
    fn clone(&self) -> Self {
        Self {
            cells: self.cells.clone(),
            keys: self.keys.clone(),
            soft: self.soft.clone(),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.cells.clone_from(&source.cells);
        self.keys.clone_from(&source.keys);
        self.soft.clone_from(&source.soft);
    }
}

impl Assignment {
    /// Assignment of every row of `x` under `t`, computed in full.
    pub(crate) fn full(x: &Data, t: &Tessellation, geometry: &Geometry) -> Self {
        let mut out = Self::default();
        Self::full_into(x, t, geometry, &mut out);
        out
    }

    /// [`full`](Self::full) written into `out`, which keeps its capacity.
    pub(crate) fn full_into(x: &Data, t: &Tessellation, geometry: &Geometry, out: &mut Self) {
        let n = x.n_rows();
        out.cells.clear();
        out.keys.clear();
        if t.tau.is_none() && geometry.is_plain() && n > 0 {
            out.soft = None;
            out.cells.reserve(n);
            nearest_blocks(x, t, |_, cells| out.cells.extend_from_slice(cells));
            return;
        }
        if t.tau.is_none() {
            out.soft = None;
            for i in 0..n {
                let (cell, key) = t.nearest(x.row(i), geometry);
                out.cells.push(cell as CellIndex);
                out.keys.push(key);
            }
            return;
        }
        let b = t.n_cells();
        let soft = out.soft.get_or_insert_with(Vec::new);
        soft.clear();
        soft.resize(b * n, 0.0);
        for i in 0..n {
            let row = x.row(i);
            let mut best = f64::INFINITY;
            let mut best_cell = 0;
            for k in 0..b {
                let key = t.key(row, k, geometry);
                soft[k * n + i] = key;
                if key < best {
                    best = key;
                    best_cell = k;
                }
            }
            out.cells.push(best_cell as CellIndex);
            out.keys.push(best);
        }
    }

    /// Row-major n by b kernel weights at bandwidth `tau`: per
    /// observation exp(-(key - winning key) / (2 tau^2)) over the
    /// centres, normalised, so the nearest centre's factor is 1.
    pub(crate) fn soft_weights(&self, tau: f64) -> Vec<f64> {
        let keys = self.soft.as_ref().expect("soft keys are cached");
        let n = self.cells.len();
        let b = keys.len() / n;
        let scale = 1.0 / (2.0 * tau * tau);
        let mut weights = vec![0.0; n * b];
        for i in 0..n {
            let min = self.keys[i];
            let mut total = 0.0;
            for k in 0..b {
                let g = crate::maths::exp(-(keys[k * n + i] - min) * scale);
                weights[i * b + k] = g;
                total += g;
            }
            for w in &mut weights[i * b..(i + 1) * b] {
                *w /= total;
            }
        }
        weights
    }

    /// The assignment under `new`, which differs from the tessellation
    /// this cache was built for by `delta`, written into `out`; equal to
    /// [`Assignment::full`] on the same inputs. With dims unchanged an
    /// untouched centre's key against any observation is unchanged, so
    /// only pairs involving the touched centre are recomputed. `scratch`
    /// is the key buffer of the streaming paths.
    ///
    /// Returns `true` when the added or moved centre is known to win no
    /// observation, so that its cell is empty; `false` promises nothing.
    pub(crate) fn updated_into(
        &self,
        x: &Data,
        new: &Tessellation,
        delta: Delta,
        geometry: &Geometry,
        scratch: &mut Vec<f64>,
        out: &mut Self,
    ) -> bool {
        let n = x.n_rows();
        match delta {
            Delta::Full => Self::full_into(x, new, geometry, out),
            Delta::CentreAdded => {
                let added = new.n_cells() - 1;
                out.clone_from(self);
                if self.soft.is_none() && geometry.is_plain() {
                    let buffer = key_buffer(scratch, 2 * n + new.n_cells());
                    let (keys, rest) = buffer.split_at_mut(n);
                    let (own, coordinate) = rest.split_at_mut(n);
                    centre_keys(x, new, added, keys);
                    // With one centre before the birth, every row's own key
                    // is that centre's key.
                    if added == 1 {
                        centre_keys(x, new, 0, own);
                    } else {
                        own_keys(x, new, &out.cells, coordinate, own);
                    }
                    return !take_nearer(keys, added, &mut out.cells, own);
                }
                if let Some(soft) = &mut out.soft {
                    soft.reserve(n);
                }
                for i in 0..n {
                    let key = new.key(x.row(i), added, geometry);
                    if let Some(soft) = &mut out.soft {
                        soft.push(key);
                    }
                    if key < out.keys[i] {
                        out.keys[i] = key;
                        out.cells[i] = added as CellIndex;
                    }
                }
            }
            Delta::CentreMoved(moved) => {
                out.clone_from(self);
                if self.soft.is_none() && geometry.is_plain() {
                    // A lone centre keeps every row wherever it moves.
                    if new.n_cells() == 1 {
                        return false;
                    }
                    let buffer = key_buffer(scratch, 2 * n + new.n_cells());
                    let (keys, rest) = buffer.split_at_mut(n);
                    let (own, coordinate) = rest.split_at_mut(n);
                    centre_keys(x, new, moved, keys);
                    own_keys(x, new, &self.cells, coordinate, own);
                    let mut won = false;
                    for (i, (&key, &own)) in keys.iter().zip(own.iter()).enumerate() {
                        let incumbent = self.cells[i] as usize;
                        if incumbent == moved {
                            let (cell, _) = nearest_plain(new, x.row(i));
                            won |= cell == moved;
                            out.cells[i] = cell as CellIndex;
                        } else if key < own || (key == own && moved < incumbent) {
                            out.cells[i] = moved as CellIndex;
                            won = true;
                        }
                    }
                    return !won;
                }
                for i in 0..n {
                    let row = x.row(i);
                    let key = new.key(row, moved, geometry);
                    if let Some(soft) = &mut out.soft {
                        soft[moved * n + i] = key;
                    }
                    let incumbent = self.cells[i] as usize;
                    if incumbent == moved {
                        let (cell, key) = new.nearest(row, geometry);
                        out.cells[i] = cell as CellIndex;
                        out.keys[i] = key;
                    } else {
                        // Strict comparison keeps the lowest-index tie rule:
                        // `moved` wins only when nearer than the incumbent
                        // or equal with a lower index.
                        let own = self.keys[i];
                        if key < own || (key == own && moved < incumbent) {
                            out.cells[i] = moved as CellIndex;
                            out.keys[i] = key;
                        }
                    }
                }
            }
            Delta::CentreRemoved(removed) => {
                out.clone_from(self);
                if let Some(soft) = &mut out.soft {
                    soft.drain(removed * n..(removed + 1) * n);
                }
                let plain = geometry.is_plain();
                let removed_index = removed as CellIndex;
                for i in 0..n {
                    if self.cells[i] == removed_index {
                        let (cell, key) = if plain {
                            nearest_plain(new, x.row(i))
                        } else {
                            new.nearest(x.row(i), geometry)
                        };
                        out.cells[i] = cell as CellIndex;
                        if !out.keys.is_empty() {
                            out.keys[i] = key;
                        }
                    } else if self.cells[i] > removed_index {
                        out.cells[i] -= 1;
                    }
                }
            }
        }
        false
    }

    /// [`updated_into`](Self::updated_into) as a fresh assignment.
    #[cfg(test)]
    pub(crate) fn updated(
        &self,
        x: &Data,
        new: &Tessellation,
        delta: Delta,
        geometry: &Geometry,
    ) -> Self {
        let mut out = Self::default();
        self.updated_into(x, new, delta, geometry, &mut Vec::new(), &mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::{chain_rng, standard_normal, uniform_index};

    fn random_tessellation(
        p: usize,
        b: usize,
        d: usize,
        rng: &mut crate::rng::Rng,
    ) -> Tessellation {
        let mut dims = Vec::new();
        while dims.len() < d {
            let c = uniform_index(p, rng);
            if !dims.contains(&c) {
                dims.push(c);
            }
        }
        let centres = (0..b * d).map(|_| 0.5 * standard_normal(rng)).collect();
        let mus = (0..b).map(|_| standard_normal(rng)).collect();
        Tessellation {
            centres,
            dims,
            mus,
            betas: Vec::new(),
            tau: None,
        }
    }

    fn random_data(n: usize, p: usize, rng: &mut crate::rng::Rng) -> Data {
        Data::new((0..n * p).map(|_| standard_normal(rng)).collect(), n, p).unwrap()
    }

    #[test]
    fn nearest_is_lowest_index_on_ties() {
        let g = Geometry::euclidean(1);
        let t = Tessellation {
            centres: vec![0.0, 0.0, 1.0],
            dims: vec![0],
            mus: vec![1.0, 2.0, 3.0],
            betas: Vec::new(),
            tau: None,
        };
        assert_eq!(t.nearest(&[0.0], &g), (0, 0.0));
        assert_eq!(t.nearest(&[0.75], &g), (2, 0.0625));
        assert_eq!(t.value_at(&[0.75], &g), 3.0);
        assert_eq!(t.n_cells(), 3);
        assert_eq!(t.n_dims(), 1);
    }

    #[test]
    fn full_equals_the_row_by_row_search() {
        let mut rng = chain_rng(23);
        for _ in 0..300 {
            let p = 1 + uniform_index(4, &mut rng);
            let d = 1 + uniform_index(p, &mut rng);
            let b = 1 + uniform_index(6, &mut rng);
            let x = random_data(1 + uniform_index(40, &mut rng), p, &mut rng);
            let g = Geometry::euclidean(p);
            let t = random_tessellation(p, b, d, &mut rng);
            let full = Assignment::full(&x, &t, &g);
            for i in 0..x.n_rows() {
                assert_eq!(full.cells[i] as usize, t.nearest(x.row(i), &g).0);
            }
        }
    }

    #[test]
    fn incremental_updates_equal_full_recompute() {
        use crate::geometry::Metric;
        let mut rng = chain_rng(11);
        for round in 0..400 {
            let p = 1 + uniform_index(4, &mut rng);
            let d = 1 + uniform_index(p, &mut rng);
            let b = 1 + uniform_index(5, &mut rng);
            let x = random_data(2 + uniform_index(30, &mut rng), p, &mut rng);
            // Even rounds Euclidean; odd rounds every column spherical on
            // one sphere, the coordinates already within the radian ranges.
            let g = if round % 2 == 0 {
                Geometry::euclidean(p)
            } else {
                Geometry::structure(&vec![Metric::Spherical { sphere: 0 }; p], p).unwrap()
            };
            let t = random_tessellation(p, b, d, &mut rng);
            let cache = Assignment::full(&x, &t, &g);

            let mut added = t.clone();
            added
                .centres
                .extend((0..d).map(|_| 0.5 * standard_normal(&mut rng)));
            added.mus.push(0.0);
            assert_eq!(
                cache.updated(&x, &added, Delta::CentreAdded, &g),
                Assignment::full(&x, &added, &g)
            );

            let moved_index = uniform_index(b, &mut rng);
            let mut moved = t.clone();
            for v in &mut moved.centres[moved_index * d..(moved_index + 1) * d] {
                *v = 0.5 * standard_normal(&mut rng);
            }
            assert_eq!(
                cache.updated(&x, &moved, Delta::CentreMoved(moved_index), &g),
                Assignment::full(&x, &moved, &g)
            );

            if b >= 2 {
                let removed_index = uniform_index(b, &mut rng);
                let mut removed = t.clone();
                removed
                    .centres
                    .drain(removed_index * d..(removed_index + 1) * d);
                removed.mus.remove(removed_index);
                assert_eq!(
                    cache.updated(&x, &removed, Delta::CentreRemoved(removed_index), &g),
                    Assignment::full(&x, &removed, &g)
                );
            }
        }
    }

    #[test]
    fn soft_incremental_updates_equal_full_recompute() {
        let mut rng = chain_rng(19);
        for _ in 0..200 {
            let p = 1 + uniform_index(4, &mut rng);
            let d = 1 + uniform_index(p, &mut rng);
            let b = 1 + uniform_index(5, &mut rng);
            let x = random_data(2 + uniform_index(30, &mut rng), p, &mut rng);
            let g = Geometry::euclidean(p);
            let mut t = random_tessellation(p, b, d, &mut rng);
            t.tau = Some(0.3);
            let cache = Assignment::full(&x, &t, &g);

            let mut added = t.clone();
            added
                .centres
                .extend((0..d).map(|_| 0.5 * standard_normal(&mut rng)));
            added.mus.push(0.0);
            assert_eq!(
                cache.updated(&x, &added, Delta::CentreAdded, &g),
                Assignment::full(&x, &added, &g)
            );

            let moved_index = uniform_index(b, &mut rng);
            let mut moved = t.clone();
            for v in &mut moved.centres[moved_index * d..(moved_index + 1) * d] {
                *v = 0.5 * standard_normal(&mut rng);
            }
            assert_eq!(
                cache.updated(&x, &moved, Delta::CentreMoved(moved_index), &g),
                Assignment::full(&x, &moved, &g)
            );

            if b >= 2 {
                let removed_index = uniform_index(b, &mut rng);
                let mut removed = t.clone();
                removed
                    .centres
                    .drain(removed_index * d..(removed_index + 1) * d);
                removed.mus.remove(removed_index);
                assert_eq!(
                    cache.updated(&x, &removed, Delta::CentreRemoved(removed_index), &g),
                    Assignment::full(&x, &removed, &g)
                );
            }
        }
    }

    /// Moved centres that tie an incumbent: the strict lowest-index rule
    /// decides, not the order of the update.
    #[test]
    fn a_moved_centre_tying_an_incumbent_takes_the_row_only_from_a_higher_index() {
        let x = Data::new(vec![0.0, 0.5], 2, 1).unwrap();
        let g = Geometry::euclidean(1);
        for (centres, moved) in [([1.0, 3.0], 1), ([3.0, 1.0], 0)] {
            let t = Tessellation {
                centres: centres.to_vec(),
                dims: vec![0],
                mus: vec![0.0, 0.0],
                betas: Vec::new(),
                tau: None,
            };
            let mut new = t.clone();
            new.centres[moved] = -1.0;
            let updated =
                Assignment::full(&x, &t, &g).updated(&x, &new, Delta::CentreMoved(moved), &g);
            assert_eq!(updated, Assignment::full(&x, &new, &g));
            assert_eq!(updated.cells[0], 0);
        }
    }

    mod props {
        use super::*;
        use proptest::prelude::*;

        /// A coordinate on a 0.25 grid, so that keys tie exactly.
        fn coordinate() -> impl Strategy<Value = f64> {
            (-4i8..=4).prop_map(|v| f64::from(v) * 0.25)
        }

        /// Data of fewer than `rows` rows and `columns` columns, a hard
        /// tessellation of `b` centres on a shuffled subset of the columns,
        /// and the index and coordinates of a moved centre.
        fn case(
            columns: usize,
            rows: usize,
        ) -> impl Strategy<Value = (Data, Tessellation, usize, Vec<f64>)> {
            (1..columns, 1usize..6, 1..rows).prop_flat_map(|(p, b, n)| {
                (1..=p).prop_flat_map(move |d| {
                    (
                        prop::collection::vec(coordinate(), n * p),
                        Just((0..p).collect::<Vec<_>>()).prop_shuffle(),
                        prop::collection::vec(coordinate(), b * d),
                        0..b,
                        prop::collection::vec(coordinate(), d),
                    )
                        .prop_map(
                            move |(values, columns, centres, k, moved)| {
                                let t = Tessellation {
                                    centres,
                                    dims: columns[..d].to_vec(),
                                    mus: vec![0.0; b],
                                    betas: Vec::new(),
                                    tau: None,
                                };
                                (Data::new(values, n, p).unwrap(), t, k, moved)
                            },
                        )
                })
            })
        }

        /// Whether no observation sits in cell `k`.
        fn empty(a: &Assignment, k: usize) -> bool {
            !a.cells.contains(&(k as CellIndex))
        }

        proptest! {
            #[test]
            fn updates_equal_the_full_recompute_under_ties(
                (x, t, k, coordinates) in case(4, 24),
            ) {
                let g = Geometry::euclidean(x.n_cols());
                let cache = Assignment::full(&x, &t, &g);
                let d = t.n_dims();
                let mut scratch = Vec::new();
                let mut out = Assignment::default();

                let mut moved = t.clone();
                moved.centres[k * d..(k + 1) * d].copy_from_slice(&coordinates);
                let known_empty = cache.updated_into(
                    &x, &moved, Delta::CentreMoved(k), &g, &mut scratch, &mut out,
                );
                let full = Assignment::full(&x, &moved, &g);
                prop_assert_eq!(&out, &full);
                prop_assert_eq!(known_empty, empty(&full, k));

                let mut added = t.clone();
                added.centres.extend_from_slice(&coordinates);
                added.mus.push(0.0);
                let known_empty = cache.updated_into(
                    &x, &added, Delta::CentreAdded, &g, &mut scratch, &mut out,
                );
                let full = Assignment::full(&x, &added, &g);
                prop_assert_eq!(&out, &full);
                prop_assert_eq!(known_empty, empty(&full, t.n_cells()));
            }

            /// Up to 599 rows, past the block boundaries, and up to six
            /// active columns, past the specialised widths.
            #[test]
            fn full_equals_the_row_by_row_search_under_ties(
                (x, t, _, _) in case(7, 600),
            ) {
                let g = Geometry::euclidean(x.n_cols());
                let full = Assignment::full(&x, &t, &g);
                for i in 0..x.n_rows() {
                    prop_assert_eq!(full.cells[i] as usize, t.nearest(x.row(i), &g).0);
                }
            }
        }
    }

    #[test]
    fn soft_weights_reproduce_the_soft_value() {
        let mut rng = chain_rng(23);
        for _ in 0..100 {
            let p = 1 + uniform_index(3, &mut rng);
            let d = 1 + uniform_index(p, &mut rng);
            let b = 1 + uniform_index(5, &mut rng);
            let n = 2 + uniform_index(20, &mut rng);
            let x = random_data(n, p, &mut rng);
            let g = Geometry::euclidean(p);
            let mut t = random_tessellation(p, b, d, &mut rng);
            let tau = 0.1 + 0.5 * standard_normal(&mut rng).abs();
            t.tau = Some(tau);
            let cache = Assignment::full(&x, &t, &g);
            let weights = cache.soft_weights(tau);
            for i in 0..n {
                let row = &weights[i * b..(i + 1) * b];
                let total: f64 = row.iter().sum();
                assert!((total - 1.0).abs() < 1e-12, "{total}");
                // The hard winner carries the largest weight.
                let winner = cache.cells[i] as usize;
                assert!(row.iter().all(|&w| w <= row[winner] + 1e-15));
                let value: f64 = row.iter().zip(&t.mus).map(|(&w, &mu)| w * mu).sum();
                let direct = t.value_at(x.row(i), &g);
                assert!((value - direct).abs() < 1e-12, "{value} vs {direct}");
            }
        }
    }

    #[test]
    fn a_tiny_bandwidth_recovers_the_hard_value() {
        let g = Geometry::euclidean(1);
        let mut t = Tessellation {
            centres: vec![0.0, 1.0],
            dims: vec![0],
            mus: vec![2.0, -3.0],
            betas: Vec::new(),
            tau: Some(1e-3),
        };
        assert!((t.value_at(&[0.1], &g) - 2.0).abs() < 1e-9);
        assert!((t.value_at(&[0.9], &g) + 3.0).abs() < 1e-9);
        // A large bandwidth approaches the unweighted mean.
        t.tau = Some(1e3);
        assert!((t.value_at(&[0.1], &g) + 0.5).abs() < 1e-4);
    }

    #[cfg(feature = "experimental")]
    #[test]
    fn serde_keeps_the_bandwidth() {
        let t = Tessellation {
            centres: vec![0.1, 0.2],
            dims: vec![0],
            mus: vec![1.0, -1.0],
            betas: Vec::new(),
            tau: Some(0.2),
        };
        let json = serde_json::to_string(&t).unwrap();
        assert!(json.contains(r#""tau":0.2"#), "{json}");
        assert_eq!(serde_json::from_str::<Tessellation>(&json).unwrap(), t);
        for bad in [
            r#"{"centres":[0.1],"dims":[0],"mus":[1.0],"tau":0.0}"#,
            r#"{"centres":[0.1],"dims":[0],"mus":[1.0],"tau":-0.2}"#,
        ] {
            assert!(serde_json::from_str::<Tessellation>(bad).is_err());
        }
    }

    #[test]
    fn serde_validates() {
        let t = Tessellation {
            centres: vec![0.1, 0.2],
            dims: vec![2],
            mus: vec![1.0, -1.0],
            betas: Vec::new(),
            tau: None,
        };
        let json = serde_json::to_string(&t).unwrap();
        assert_eq!(serde_json::from_str::<Tessellation>(&json).unwrap(), t);
        for bad in [
            r#"{"centres":[0.1],"dims":[2],"mus":[1.0,-1.0]}"#,
            r#"{"centres":[0.1,0.2],"dims":[2,2],"mus":[1.0]}"#,
            r#"{"centres":[],"dims":[],"mus":[]}"#,
        ] {
            assert!(serde_json::from_str::<Tessellation>(bad).is_err());
        }
    }
}
