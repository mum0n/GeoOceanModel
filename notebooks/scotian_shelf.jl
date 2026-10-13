### A Pluto.jl notebook ###
# v1.0.4

using Markdown
using InteractiveUtils

# ╔═╡ 3d1a0c52-7b21-4f6e-9c2b-6f6f4a0a5d01
# ╠═╡
using Pkg

# ╔═╡ 2f9a1e77-5c34-4b8d-9a71-1c2f8b3e5a10
# ╠═╡
md"""
# Scotian Shelf ocean model — the production configuration

Work with the `scotian_shelf` domain end to end: resolve the seabed through the GeoData layer,
build the immersed grid, assemble the hydrostatic model, run the simulation, and read the
resulting GeoData store back.

Every input is a named, published product piped in through **GeoData** (`GeoDataSources`), and
every output lands in the shared lakehouse. Nothing here is synthetic stand-ins: if a download
fails, the run stops and says so, rather than quietly running on different data.

### What this notebook documents

Each step is an ordinary package call, so a cell can be re-run on its own while you inspect what
it returned — that is what Pluto buys you over a script. The markdown between the cells is the
documentation: **what each option means, which assumptions the model bakes in, and how each
physical process is parameterized**. The TOML comments say the same things in fewer words;
here the same choices are explained with the resolved value in front of you.

### Runtime

`scotian_shelf` ships a **3-day** (72 h) run at a 1800 s output cadence — 145 intervals. The
first interval compiles the turbulent-closure kernels, which dominates the wall clock; after
that it is minutes. Raise `[hydrodynamics] sim_duration_hours` for a longer run, and note that
an existing checkpoint is picked up automatically when `auto_restart = true`.

This configuration needs a Copernicus Marine account for its open boundaries — see
[_Which inputs need an account_](#) below for which input that is and what happens without it.
"""

# ╔═╡ 1b3c9d84-6a52-4c0e-8f4b-2a7d5e9c0f33
# ╠═╡
md"""
## Running this notebook

Launch Pluto from an environment that already has it (the global environment, or one where you
have added `Pluto`) and open this file from the notebook list:

```
julia -e 'using Pluto; Pluto.run()'
```

The first three cells pin the working directory, activate **this package's environment**, and
load the package, so the `../geodatalakehouse/...` paths in the configs resolve to
`projects/lakehouse/` rather than to a directory inside the package, and `using GeoOceanModel`
loads the working tree rather than a registered version.

`Pluto` is deliberately **not** a dependency of this package: a notebook is how you look at the
work, not something the run needs. Its version and HTTP server have historically conflicted with
this package's pinned `Zarr`/`YAXArrays` stack, so it is installed only where the notebooks run.

One consequence of the format is worth knowing, because it is what makes this notebook useful:
a cell defines something and every later cell sees it, and re-running a cell re-runs everything
downstream of it. So changing `sim_duration_hours` and re-running lets you watch the store's
interval count move, without re-running the parts that have not changed.
"""

# ╔═╡ a0c1f6b3-2d84-4e79-9b0c-3f5d7a1e6c44
# ╠═╡
# Pin the working directory to the package root: otherwise the `../lakehouse/...` paths in the
# configs resolve relative to wherever the notebook process started.
cd(joinpath(@__DIR__, ".."))

# ╔═╡ c4e8b2a1-9f3d-4c6b-8e15-7d0a2b9f5c18
# ╠═╡
# Activating the package directory from inside the notebook means Pluto's own environment never
# needs this package, and `using GeoOceanModel` sees the working tree. A Pluto save of this file
# stores the activate call, so re-opening it reproduces the same environment.
Pkg.activate(@__DIR__)

# ╔═╡ 5e2f7d09-4a16-4b83-9c1d-8b3e6f0a1d27
# ╠═╡
using GeoOceanModel

# ╔═╡ 6b8a3c47-1d29-4f0e-8a56-9c4d2e7b0f31
# ╠═╡
md"""
## Which run this is, and what it needs

`configs/scotian_shelf.toml` — read the same way any config is read. This is the production
configuration: the snow-crab study domain, **345 × 245 × 13** at ~2.5 km horizontal resolution,
run for **72 simulated hours** at an **1800 s output cadence** (145 intervals).

### Which inputs need an account

Only **one**: the Copernicus Marine download for the GLORYS12V1 open-boundary water and initial
hydrography. It is free, but it wants registration, and then reads a username and key from
`~/.copernicusmarine/credentials`. Without that file the download is refused and the run stops —
it does **not** fall back to a coarser product, because the open edges would then hold different
water and nothing in the output would show it. Everything else this configuration uses — NWA
measured bathymetry, Natural Earth coastline, Open-Meteo winds, WOA23 climatology, TPXO9 tides —
is keyless, and the sections below name each one where it enters a cell.
"""

# ╔═╡ 7c9d4e58-2b3f-4a01-9d67-0e5b3f8c1a42
# ╠═╡
const RUN = "scotian_shelf"

# ╔═╡ 8d5e6f09-3c40-4b12-8e78-1f6c4a9d2b53
# ╠═╡
md"""
## Configuration: reading the TOML, then overriding where it writes

`load_configuration` reads the TOML and validates its sections, so a misspelled section fails
here rather than silently defaulting later. `configuration_to_options` turns the dictionary into
the flat `HydrodynamicOptions` record every segment consumes. The two-step split matters: the
first is data you can inspect and edit, the second is what the code actually runs on.

### Where the run writes

Everything this run writes — the fetched bathymetry, coastline and winds, the GeoData store, and
the JLD2 checkpoint with its checkpoints — goes to **`OUTDIR`**. Point it anywhere: an absolute
path, or one relative to the package root. A scratch run outside the shared lakehouse costs disk
and buys a self-contained directory you can delete; writing inside the lakehouse shares the
fetched products with every later run.

The override is applied to the configuration dictionary **before** the options are built, so no
TOML file is ever edited:

```julia
cfg["storage"]["output_path"] = OUTDIR
opts = configuration_to_options(cfg)
```

`storage.output_path` is the run's directory; `input_dir` defaults to `<output path>/inputs`, and
the bathymetry, coastline and wind artefacts all land there.
"""

# ╔═╡ 9e7f0a19-4d51-4c23-9f89-2a7d5b0e3c64
# ╠═╡
import GeoOceanModel: load_configuration, configuration_to_options

const OUTDIR = normpath(joinpath("..", "lakehouse", "GeoOceanModel", RUN))

cfg = load_configuration(joinpath("configs", string(RUN, ".toml")))
cfg["storage"]["output_path"] = OUTDIR
opts = configuration_to_options(cfg)

println("config   : ", joinpath("configs", string(RUN, ".toml")))
println("domain   : lon ", opts.domain_lon, "  lat ", opts.domain_lat)
println("z range  : ", opts.domain_z, " m")
println("grid     : ", opts.grid_size, " cells (Nx, Ny, Nz)")
println("duration : ", opts.sim_duration / 3600, " hours (", opts.sim_duration / 86400, " days)")
println("OUTDIR   : ", opts.output_dir)
println("inputs   : ", opts.input_dir)

# ╔═╡ af081b2a-5e62-4d34-9a90-3b8e6c1d4e75
# ╠═╡
md"""
## Where the files live: the lakehouse

Two stores, and they are separate on purpose: the shared central registry on the left, and this
run's own directory on the right.

```
projects/
├── GeoOceanModel/            this package: grid, physics, ocean figures
│   └── configs/*.toml        the shipped run definitions
├── GeoData/                  the abstraction layer: geoload / geosave / landmask
├── GeoDataSources/           the fetchers: what gets downloaded, and where it is cached
└── lakehouse/
    ├── GeoData/              the central assimilated caches: woa23, tpxo9, glorys,
    │                         etopo2022, era5 — fetched once per machine, not once per run
    └── GeoOceanModel/
        └── scotian_shelf/    this run's directory
            ├── inputs/       the products this run fetched: bathymetry, coastline, winds
            ├── checkpoints/  Oceananigans state checkpoints (restart from here)
            └── hydrodynamics.zarr   the realized flow, as a GeoData store
```

The run reads the **central** copy first. That is the point of the lakehouse: a ~1.8 GB
climatology or tidal atlas is a machine-level asset, so a second run reuses it instead of
downloading it again. It also means two runs of the *same* config share an input, and two runs of
*different* configs do not — each config names its own products, so a stale cached copy is a
visible difference rather than a silent one.

The store this notebook ends with — `hydrodynamics.zarr` — is a self-contained GeoData directory.
The coordinates live inside its attributes alongside `time.json`, so the file is the handover;
nothing is serialized through `SimulationResult` on this path.
"""

# ╔═╡ b129c03b-6f73-4e45-9b01-4c9f7d2e5f86
# ╠═╡
md"""
## The data segment: what gets fetched, and why each product

`run_segment_data` is the ingestion step: coastline, bathymetry, wind, and the boundary and
initial hydrography, each named once in the TOML. The bathymetry call below goes through the
same layer, so the sea floor resolved here is the one the immersed grid is built from — every
later segment sees the same one, because they all ask through here.

### `[data]` — the four fetched products

| key | value here | what it is | account? |
|---|---|---|---|
| `bathy_source` | `nwa_bathymetry` | measured Northwest Atlantic shelf soundings — the right product for this shelf, not a global model | no |
| `coastline_source` | `natural_earth_10m` | public-domain land outline at full detail; `50m`/`110m` are coarser | no |
| `wind_source` | `open_meteo` | real past weather through a free API; also air temperature, pressure, humidity, cloud and sunlight, so the heat budget can be closed | no |
| `hydrography_source` | `glorys12v1` | the reanalysis that seeds the initial T/S | yes |

There is **no fallback dataset**: ETOPO 2022 is not offered as a backstop for the NWA product, so
an unavailable NWA fails loudly instead of silently running on coarser relief. Two alternatives
were considered and set aside, and the reasons are recorded in the TOML: NOAA's own reanalysis
stores its numbers in a layout with no lat/lon labels (getting that wrong displaces the wind a
whole degree with no visible error), and the Global Wind Atlas hands over a picture file needing
extra software.

### `[bathymetry]`

- `resolution_arcsec = 15` — ~500 m at mid-latitudes. Beware: resolution is the single biggest
  driver of fetch size and of grid construction cost here.
- `inshore_depth = -20.0` — the depth the landward edge is pinned to, so the shoreline is a wet
  cell rather than a cliff.
- `shelf_slope = 500.0` — the slope scale the vertical filter smooths against.
- `mask_bay_of_fundy = true` — hypertidal shallows set to solid land. That is a deliberate
  exclusion: they generated spurious vertical velocities (~1.3 m/s) and pinned the adaptive
  timestep to ~1 s, and snow crab do not enter the bay.
- `ingest_library = "geodata"` — which library performs the ingestion and regridding.

### `[boundaries]` — the open edges

The domain is a rectangle cut out of a larger sea, so three edges are imaginary lines where water
crosses. `ocean_boundary_source = "glorys12v1"` constrains the temperature, salinity **and**
velocity just outside. WOA23 was the keyless alternative and was set aside for one reason: it
carries no currents, and for a dispersal calculation the volume coming through the edge is the
thing that matters. `method = "relaxation"` (Price & Aumont 2011) eases the edge band toward the
outside conditions over `sponge_timescale_seconds = 3600`, across
`sponge_layer_width = 0.25` degrees — blending, not snap-forcing, so the interior stays free.
"""

# ╔═╡ c23d104c-7f84-4e56-9c12-5d0f8e3a6b97
# ╠═╡
import GeoOceanModel: run_segment_data, _run_bathymetry

data_res = run_segment_data(; opts = opts)

_coast = joinpath(opts.input_dir, "coastline.parquet")
println("\ncoastline : ", _coast, " (", isfile(_coast) ? "present" : "absent", ")")
println("bathymetry: ", data_res.bathy_file)
println("wind      : ", data_res.wind_file)
println("tau_x/y   : ", round(data_res.tau_x, digits = 6), ", ", round(data_res.tau_y, digits = 6), " m²/s²")

bathy = _run_bathymetry(; opts = opts, force = false)

println("\nsource   : ", bathy.source)
println("artefact : ", bathy.output_path)
println("elevation: ", extrema(bathy.elevation), " m")

wet = count(x -> x < 0, bathy.elevation)
println("wet cells: ", wet, " / ", length(bathy.elevation), "  (", round(100 * wet / length(bathy.elevation), digits = 1), "%)")

# ╔═╡ d34e215d-8a95-4f67-9d23-6e1f9f4b7c08
# ╠═╡
md"""
## The grid: three vertical segments in one column

`build_shelf_grid` (reached through `run_segment_grid`) places the cells: `nx × ny` horizontally
over `[domain] lon_min..lon_max`, `lat_min..lat_max`, and `nz` layers down the vertical faces.
The segment prints a summary of what it resolved; the cell below re-reads it from the returned
grids so you can see the same numbers from the outside.

The interesting decision is vertical, and it is not obvious from the numbers:

- `depths` lists **14 faces** for `nz = 13` cells — an explicit discretisation, positive downward,
  which must span `z_min`. The faces are negated and reversed because the config writes them
  positive-down and Oceananigans wants them negative and ascending; `nz` must therefore equal
  `length(depths) - 1`, and the segment refuses to proceed when the two disagree rather than
  quietly picking one.
- An explicit depth list **takes precedence over** `vertical_stretching_mode`. This config sets no
  mode at all, so the package default (`"oceananigans"`, evenly spaced faces) would have applied
  if the list were removed.
- The two remaining keys (`nz_above = 0`, `vertical_break_depth = -400.0`) belong to the
  `"two_segment"` mode and are **unused** while `depths` is set. They are kept in the file
  because switching to that mode is a one-line change: drop the depth list, set
  `vertical_stretching_mode = "two_segment"`, and the column splits at the break depth with a
  surface-refined segment above and a coarse one below, the split derived as 60% of `nz`.
- The faces are **not evenly spaced**: they cluster in the 0–400 m active layer and stretch to
  5000 m to cover the Laurentian Fan cheaply. 22% of this domain is deeper than 400 m, so a
  surface-refined-only column would have to be enormous.

The sampling points that matter downstream are the grid's **interior cell centres**, not the face
extent. Getting that window wrong is what once put the sea floor half a cell off; the driver
resolves them through `_grid_interior_centres` so the interpolation and the immersion agree.

### `[domain]` and `[runtype]`

`buffer_km = 50.0` is a margin this run does not use; it exists for consumers that seed particles.
`use_gpu = true` with `fallback_to_cpu = false` means a run whose GPU request cannot be honoured
fails loudly instead of quietly running somewhere other than where the config said it would.
`seed = 42` fixes the RNG so two runs of the same config differ only by the physics.
"""

# ╔═╡ e45f326e-9ba6-4f78-9d34-7f2a0b5c8d19
# ╠═╡
import GeoOceanModel: run_segment_grid

grid_res = run_segment_grid(; opts = opts)

bg = grid_res.base_grid
println("base grid: Nx ", bg.Nx, "  Ny ", bg.Ny, "  Nz ", bg.Nz)
println("lon span : ", bg.λᶠᵃᵃ[1], "°E to ", bg.λᶠᵃᵃ[end], "°E")
println("haloes   : Hx ", grid_res.immersed_grid.Hx, "  Hy ", grid_res.immersed_grid.Hy)
println("immersed : ", hasproperty(grid_res.immersed_grid, :immersed_boundary))

# The wet fraction of the sea floor, from the host array the immersion was regridded from. The
# immersed seabed itself is a halo-padded OffsetArray over a CuArray on a GPU run, and neither
# `count` with a predicate nor broadcasting over it is allowed there.
wet = count(x -> x < 0, bathy.elevation)
println("wet seabed: ", wet, " / ", length(bathy.elevation), " cells (",
        round(100 * wet / length(bathy.elevation), digits = 1), "%)")

# ╔═╡ f560437f-0cb7-4e89-9e45-8a3b1c6d9e2a
# ╠═╡
md"""
## The model: what it is, and what it assumes

`run_segment_model` assembles a `HydrostaticFreeSurfaceModel` out of that grid and the fetched
inputs. This is the slow cell — the first interval compiles the turbulent-closure kernels — so
run it alone if you want to inspect the grid before paying for a simulation.

### Parameterizations, one by one

**Equation of state — TEOS-10**, not the linearised form, with reference density 1025 kg m⁻³
rather than the library's 1020 default. Buoyancy and N² therefore differ from older linear
results: an accuracy change, not a like-for-like swap, so do not compare numbers across the
change.

**Coriolis — spherical** (`HydrostaticSphericalCoriolis`), so the β term survives. Freezing f at
a single latitude would bias the vorticity balance over a 42–47.5 °N domain five degrees wide.
`coriolis_latitude = 44.5` in the TOML is only a reference value for the scheme's constants.

**Free surface — split-explicit**: barotropic waves are subcycled rather than solved implicitly.
Cheaper than an implicit free surface, and the library's standard choice for shelf-scale work.

**Vertical turbulence — `turbulence_closure = "nemotke"`**, the prognostic NEMO-TKE scheme: a
TKE budget is carried as an extra field and mixed lengths are diagnosed from it. It costs one
more tracer and a stricter step limit, and it buys stratification that actually responds to the
wind instead of being prescribed.

**Horizontal diffusion — fixed `HorizontalScalarDiffusivity(ν = κ = 100.0)`**, hard-coded in
`build_hydrodynamic_model`. This is numerical smear to keep the advection scheme stable, **not a
calibrated eddy diffusivity**, and it is deliberately not exposed in the TOML so it cannot be
mistaken for a tuned parameter.

**Bottom drag — quadratic, Patankar-implicit.** `[bottom_boundary_layer] cd_drag = 0.0025` is the
quadratic coefficient, `bottom_drag = 0.0001` s⁻¹ a background linear rate, and
`bbl_mixing_closure = "tke"` a benthic TKE closure. Implicit treatment is what makes a large
coefficient usable: it cannot stiffen the momentum equation.

**Surface heat — a constant 50 W m⁻²**, because `bulk_heat_flux = false`. The reason is recorded
in the TOML: `build_bulk_surface_flux` needs the sea-surface temperature, and a surface boundary
condition cannot currently be given a tracer dependency in this Oceananigans version. So the run
uses a fixed flux rather than one that reads back the temperature it just changed. That is a
real limitation, not a simplification of convenience.

**Wind drag — `drag_formulation = "garratt_1977"`** over `u10, v10` (plus `t2m`, `ssrd`, `strd`
when bulk fluxes are enabled). Stronger wind means more drag, through the standard bulk
coefficient.

**Tides — TPXO9**, `constituents = ["M2", "S2"]`, i.e. the two that make the spring–neap cycle.
The atlas gives a tidal *velocity*, and a velocity cannot be fed to a momentum `Forcing`, which
wants an acceleration — doing so diverged the run. The tide is instead **relaxed toward** over
`tidal_relaxation_timescale_seconds = 3600`, which is the correct expression of a prescribed
oscillating velocity and is a physics choice: small pins the flow to the tide and suppresses wind
and buoyancy, large lets both contribute.

**Sea ice and oxygen — on.** `enable_sea_ice` adds thermodynamic freezing and frazil
diagnostics; `enable_oxygen` carries O₂ as a prognostic tracer, which WOA23 can seed.

### Assumptions worth knowing before reading the output

- **The open edges are constrained but not closed.** The sponge eases toward GLORYS12V1, so
  incoming water carries real properties; the domain is still not a periodic or a closed basin.
- **Tides are on here, off in `default.toml`.** With them off, what remains is the wind- and
  buoyancy-driven part of a shelf where velocities are tidal-dominated — a large fraction of the
  real signal removed.
- **Initial stratification comes from the reanalysis**, so the first day is a relaxation toward
  balance rather than an equilibrium.
- **Salinity is not stored.** The output callback writes u, v, w and T, so density read back from
  the store is derived from temperature alone against the reference 35 PSU — an approximation,
  and the reason `compute_hydrodynamic_diagnostics` treats salinity as optional.
- **`sim_dt = 60` s is only the initial step.** `adaptive_cfl = true` with target 0.5 and a
  3600 s ceiling means the step is set dynamically; the reported step during the run will not be
  60 s.
"""

# ╔═╡ 06715480-1dc9-4a01-9b12-9a4e2f7b0c58
# ╠═╡
import GeoOceanModel: run_segment_model

built = run_segment_model(; opts = opts, immersed_grid = grid_res.immersed_grid)

println("Simpson-Hunter, shallow bank : ", built.simpson_hunter_bank)
println("Simpson-Hunter, deep shelf    : ", built.simpson_hunter_shelf)
built

# ╔═╡ 17826591-2e0a-4b12-9c23-ab5f308c1d69
# ╠═╡
md"""
## Run the simulation, and what it writes

`run_segment_simulation` advances the model and writes the flow into the GeoData store. On the
CLI, `--all` runs exactly these cells plus the data and grid segments — without the climate
perturbation, which runs only when `--climate` is named.

### The output cadence is its own decision

`[storage] output_schedule_seconds = 1800` — one field snapshot every half hour, independent of
`checkpoint_schedule_seconds = 3600`. The two were once the same knob; separating them was the
point, because a 6-hour field cadence over a 30-day run is 121 intervals, whereas a checkpoint
every 6 hours over the same run restarts from a state up to 6 hours stale. Cadence decides how
much disk the run uses and how smooth an animation looks; checkpoint cadence decides how much
work a crash loses — with `auto_restart = true` a re-run picks that state up.

### What lands in the store

- **u, v, w, T per interval** — the store keeps the model's native stagger, and the reader blends
  it to cell centres (so a `u` array is not simply the `u` that was written).
- **the grid's coordinates** travel inside the store's attributes, alongside `time.json`. That is
  what lets the handover be a directory with nothing beside it.
- **no salinity**: density read back is a thermal approximation, as above.

Re-running with a longer `sim_duration_hours` resumes from the checkpoint when
`auto_restart = true`, rather than starting over.
"""

# ╔═╡ 28937602-3f1b-4c23-9d34-bc60419d2e7a
# ╠═╡
import GeoOceanModel: run_segment_simulation

sim_res = run_segment_simulation(; opts = opts, model = built.model)

println("flow solution: ", sim_res.output_path)
sim_res

# ╔═╡ 39348713-402c-4d34-9e45-cd71520e3f8b
# ╠═╡
md"""
## Read the store back

`extract_hydrodynamic_dataset` takes the run's output **directory** — the thing
`run_segment_simulation` hands back — and reads that store. It is the same path a figure takes,
so what is read here is what gets drawn.

Ask for the **last** interval: the first is the rest state before the wind has acted, and a run
that has not yet spun up looks plausible and is not.

Three checks that mean "this is the real run and not the synthetic fallback":

1. the interval count matches the cadence (`duration / output_schedule_seconds + 1`),
2. `maximum(abs, ds.u)` is non-zero — identically zero over every interval means still water,
3. `T` spans a physical range rather than a constant — a field with no relation to the seabed
   above means the analytic synthesis fired instead of the store.
"""

# ╔═╡ 4a359824-513d-4e45-9f56-de82631f4a9c
# ╠═╡
import GeoOceanModel: extract_hydrodynamic_dataset, output_zarr_path
using Zarr, JSON3

store_dir = output_zarr_path(sim_res.output_path)
_times = Float64.(JSON3.read(read(joinpath(store_dir, "time.json"), String), Vector{Float64}))

println("store     : ", store_dir)
println("intervals : ", length(_times), " (expected ", opts.sim_duration / opts.output_schedule_seconds + 1, ")")
println("last at   : ", _times[end] / 86400, " days")

ds = extract_hydrodynamic_dataset(sim_res.output_path; time_seconds = last(_times))

println("grid     : ", length(ds.lons), " x ", length(ds.lats), " x ", length(ds.depths))
println("u        : ", extrema(ds.u))
println("v        : ", extrema(ds.v))
println("T        : ", extrema(ds.temperature))

# ╔═╡ 5b46a935-624e-4f56-9a67-ef93742050ad
# ╠═╡
# Real flow, not zeros and not synthesis.
maximum(abs, ds.u)

# ╔═╡ 6c57ba46-735f-4e67-9a78-f04853161be
# ╠═╡
md"""
## Diagnostics: what the reader derives

`compute_hydrodynamic_diagnostics` takes the four stored fields and derives the rest:
temperature- and salinity-driven density (the latter degrading to the thermal term when salinity
is absent, which here it is), the resulting stratification, Richardson number, vorticity, and the
turbulent viscosity and diffusivity from the TKE closure. It is called by every figure, so what a
figure labels "density" is this number.

Print one profile rather than a volume: a single column is where stratification is legible.
"""

# ╔═╡ 7d68cb57-846f-4e78-9b89-15964272c1f
import GeoOceanModel: compute_hydrodynamic_diagnostics
using Statistics

diag = compute_hydrodynamic_diagnostics(ds.lons, ds.lats, ds.depths, ds.u, ds.v, ds.w, ds.temperature, ds.salinity)

imap = argmin(abs.(ds.lons .- mean(ds.lons)))
jmap = argmin(abs.(ds.lats .- mean(ds.lats)))

println("density at the column centre (kg m⁻³), surface → bottom:")
for k in eachindex(ds.depths)
    println(rpad(round(ds.depths[k], digits = 1), 9), " m  →  ", round(diag.density[imap, jmap, k], digits = 3))
end

# ╔═╡ 8e79dc68-9579-4a01-9b90-26a75383c2f
# ╠═╡
md"""
## Plot a slice

Surface speed and temperature, land masked out. White cells are the emerged land from the seabed
resolved above, which is what makes the coastline visible instead of a blank rectangle.

The mask is drawn against **the run's own fetched coastline** — read from
`<output_dir>/inputs/coastline.parquet` — not against a bundled regional outline. A figure masked
against different land than the model ran on draws land in the water and says nothing.
"""

# ╔═╡ 9f80ed79-0a29-4b12-9c01-37b86493093
# ╠═╡
using CairoMakie

# The surface cell centre, chosen by depth rather than by position: `ds.depths` runs from the
# deepest centre to the shallowest, and assuming an order is how a figure ends up plotting the
# seabed.
k_surface = argmin(abs.(ds.depths))
surface_speed = hypot.(ds.u[:, :, k_surface], ds.v[:, :, k_surface])
surface_temp = ds.temperature[:, :, k_surface]

fig = Figure(size = (960, 400))

ax_speed = Axis(fig[1, 1], title = "surface |u, v|  (m/s)", xlabel = "lon", ylabel = "lat")
hm_speed = heatmap!(ax_speed, ds.lons, ds.lats, surface_speed';
                   colormap = :thermal, nan_color = :white)
Colorbar(fig[1, 2], hm_speed, label = "m/s")

ax_temp = Axis(fig[1, 3], title = "surface temperature (degC)", xlabel = "lon", ylabel = "lat")
hm_temp = heatmap!(ax_temp, ds.lons, ds.lats, surface_temp';
                  colormap = :thermometer, nan_color = :white)
Colorbar(fig[1, 4], hm_temp, label = "degC")

fig

# ╔═╡ a091fe8a-13ab-4c23-9d12-48c97541084
# ╠═╡
md"""
## What to change, and what each change does

Knobs worth turning first, in the order they matter:

- **`[hydrodynamics] sim_duration_hours`** — the run length. Re-running resumes from the
  checkpoint, so this is cheap to extend rather than restart.
- **`[storage] output_schedule_seconds`** — how many intervals the store holds. 1800 s is
  animation-grade; 21600 s (6 h) is analysis-grade and a third of the disk.
- **`[tides] enable_tides`** — off gives the wind- and buoyancy-only circulation, on gives the
  tidal-dominated one. Change it and the field changes character, which is the point.
- **`[boundaries] ocean_boundary_source`** — `"woa23"` is the keyless constraint (temperature and
  salt, no currents); `"synthetic"` leaves the edges entirely unconstrained.
- **`[data] bathy_source`** — `"nwa_bathymetry"` is measured soundings; `"etopo2022"` is a global
  model at ~450 m and a different sea floor, so it is not a drop-in substitute.
- **`[hydrodynamics] surface_heat_flux` / `turbulence_closure`** — the two physics knobs with no
  data dependency. `turbulence_closure = "nemotke"` is the expensive one.
- **`[grid] depths`** — an explicit face list. Add faces where you want centres; the 0–50 m band
  is the one this shelf's near-surface stratification lives in.
"""

# ╔═╡ b1a20f9b-24bc-4d34-9e23-59d08652a95
# ╔═╡ Cell order:
# ╠═3d1a0c52-7b21-4f6e-9c2b-6f6f4a0a5d01
# ╠═2f9a1e77-5c34-4b8d-9a71-1c2f8b3e5a10
# ╠═1b3c9d84-6a52-4c0e-8f4b-2a7d5e9c0f33
# ╠═a0c1f6b3-2d84-4e79-9b0c-3f5d7a1e6c44
# ╠═c4e8b2a1-9f3d-4c6b-8e15-7d0a2b9f5c18
# ╠═5e2f7d09-4a16-4b83-9c1d-8b3e6f0a1d27
# ╠═6b8a3c47-1d29-4f0e-8a56-9c4d2e7b0f31
# ╠═7c9d4e58-2b3f-4a01-9d67-0e5b3f8c1a42
# ╠═8d5e6f09-3c40-4b12-8e78-1f6c4a9d2b53
# ╠═9e7f0a19-4d51-4c23-9f89-2a7d5b0e3c64
# ╠═af081b2a-5e62-4d34-9a90-3b8e6c1d4e75
# ╠═b129c03b-6f73-4e45-9b01-4c9f7d2e5f86
# ╠═c23d104c-7f84-4e56-9c12-5d0f8e3a6b97
# ╠═d34e215d-8a95-4f67-9d23-6e1f9f4b7c08
# ╠═e45f326e-9ba6-4f78-9d34-7f2a0b5c8d19
# ╠═f560437f-0cb7-4e89-9e45-8a3b1c6d9e2a
# ╠═06715480-1dc9-4a01-9b12-9a4e2f7b0c58
# ╠═17826591-2e0a-4b12-9c23-ab5f308c1d69
# ╠═28937602-3f1b-4c23-9d34-bc60419d2e7a
# ╠═39348713-402c-4d34-9e45-cd71520e3f8b
# ╠═4a359824-513d-4e45-9f56-de82631f4a9c
# ╠═5b46a935-624e-4f56-9a67-ef93742050ad
# ╠═6c57ba46-735f-4e67-9a78-f04853161be
# ╠═7d68cb57-846f-4e78-9b89-15964272c1f
# ╠═8e79dc68-9579-4a01-9b90-26a75383c2f
# ╠═9f80ed79-0a29-4b12-9c01-37b86493093
# ╠═a091fe8a-13ab-4c23-9d12-48c97541084
# ╠═b1a20f9b-24bc-4d34-9e23-59d08652a95
