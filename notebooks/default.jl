### A Pluto.jl notebook ###
# v1.0.4

using Markdown
using InteractiveUtils

# ╔═╡ d7f2c0a4-9e18-4b63-8f52-0a1c5e9b3d64
# ╠═╡
using Pkg

# ╔═╡ 1e0b94f7-6c82-4d51-9b7a-3f8e2c0a6b15
# ╠═╡
md"""
# Cape Breton ocean model — the keyless demonstration

Work with the `default` domain end to end: resolve the seabed through the GeoData layer, build the
immersed grid, assemble the hydrostatic model, run the simulation, and read the resulting GeoData
store back.

Every input here is keyless: ETOPO 2022 bathymetry, WOA23 climatology, Open-Meteo winds,
unconstrained open edges, tides off. Nothing needs registration, and the whole thing finishes in
minutes after compilation.

Where a choice is a simplification rather than physics, this notebook says so at the step where it
enters, so the two assumptions most likely to mislead you are stated where they apply rather than
listed here.

### Runtime

`default` runs **72 simulated hours** (3 days) at a 21600 s output cadence — 13 intervals. The
first interval compiles the turbulent-closure kernels and dominates the wall clock.
"""

# ╔═╡ 2a1c05e8-7d39-4e72-9c8b-4a9f3d1b7c26
# ╠═╡
md"""
## Running this notebook

Launch Pluto from an environment that already has it (the global environment, or one where you
have added `Pluto`) and open this file from the notebook list:

```
julia -e 'using Pluto; Pluto.run()'
```

The first three cells pin the working directory, activate **this package's environment**, and load
the package, so the `../lakehouse/...` paths in the configs resolve to `projects/lakehouse/` rather
than to a directory inside the package, and `using GeoOceanModel` loads the working tree.

`Pluto` is deliberately **not** a dependency of this package: a notebook is how you look at the
work, not something the run needs.

One consequence of the format is worth knowing, because it is what makes this notebook useful: a
cell defines something and every later cell sees it, and re-running a cell re-runs everything
downstream of it. So changing `sim_duration_hours` and re-running lets you watch the store's
interval count move, without re-running the parts that have not changed.
"""

# ╔═╡ b3d2e6f9-84a1-4b93-9a0c-5b4f3e2c8d37
# ╠═╡
# Pin the working directory to the package root: otherwise the `../lakehouse/...` paths in the
# configs resolve relative to wherever the notebook process started.
cd(joinpath(@__DIR__, ".."))

# ╔═╡ c4e3f70a-95b2-4ca4-9b1d-6c5f4a3d9e48
# ╠═╡
# Activating the package directory from inside the notebook means Pluto's own environment never
# needs this package, and `using GeoOceanModel` sees the working tree.
Pkg.activate(@__DIR__)

# ╔═╡ d5f4081b-0a63-4db5-9c2e-7d6f5b4e0f59
# ╠═╡
using GeoOceanModel

# ╔═╡ e60a192c-1b74-4ec6-9d3f-8e7f6c5f1a6b
# ╠═╡
md"""
## Which config, and what it assumes

`configs/default.toml`, read the same way any config is read. Three things define it:

1. **small and coarse** — a 32 × 38 × 8 grid over a 1.6° × 1.9° box, ~5 km cells. Big enough that
   the physics is real, small enough that a 3-day run is minutes.
2. **every input keyless** — no Copernicus Marine account, no tidal atlas. The products are ETOPO
   2022 (a global sea-floor *model*), WOA23 climatology for the initial temperature and salt, and
   Open-Meteo winds.
3. **the two big forcings are simplified** — the open edges are unconstrained and tides are off.
   Both are documented where they enter below, because they are the assumptions most likely to
   mislead you when reading a number off this run.
"""

# ╔═╡ f71b2a3d-2c8f-4fd7-9e4f-9f8f7d6e2b7c
# ╠═╡
const RUN = "default"

# ╔═╡ 082c3b4e-3d90-4e18-9a5f-0a9f8e7f3c8d
# ╠═╡
md"""
## Configuration

`load_configuration` reads the TOML and validates its sections; `configuration_to_options` turns
the dictionary into the flat `HydrodynamicOptions` record every segment consumes. The two-step
split matters: the first is data you can inspect and edit, the second is what the code runs on.

`OUTDIR` below is where the run writes — the fetched bathymetry, coastline and winds, the GeoData
store, and the checkpoint. It is written into the configuration **before** the options are built,
so no TOML is ever edited. Point it at an absolute path to keep a scratch run out of the shared
lakehouse.
"""

# ╔═╡ 193d4c5f-4e01-4f29-9b60-1b0f9f8a4d9e
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
println("duration : ", opts.sim_duration / 3600, " hours")
println("OUTDIR   : ", opts.output_dir)
println("inputs   : ", opts.input_dir)

# ╔═╡ 2a4e5d60-5f12-4e3a-9b71-2c1f0a9b5e0f
# ╠═╡
md"""
## The data segment: fetched products, all keyless

`run_segment_data` is ingestion: coastline, bathymetry, wind, and the initial hydrography, each
named once in the TOML.

| key | value here | what it is | account? |
|---|---|---|---|
| `bathy_source` | `etopo2022` | global modelled topography, 60 arcsec here — a *model*, and the coarser of the two products the package accepts | no |
| `coastline_source` | `natural_earth_10m` | public-domain land outline at full detail | no |
| `wind_source` | `open_meteo` | real past weather through a free API | no |
| `hydrography_source` | `woa23` | NOAA monthly mean temperature and salt; keyless | no |
| `ocean_boundary_source` | `synthetic` | the open edges are not constrained at all | no |

`etopo2022` is a global sea-floor *model*. A run on it is a run on a different sea floor than one
on measured soundings, so a bathymetry-sensitive number here is not a property of this domain —
it is a property of this domain on this relief.

### The two deliberate simplifications

- **Open edges unconstrained** (`ocean_boundary_source = "synthetic"`). The sponge band is still
  applied as geometry, so the edges do not reflect, but incoming water carries no prescribed
  properties. This is the single largest simplification in this config.
- **Tides off** (`enable_tides = false`, `tides_source = "analytic"`). On a shelf where velocities
  are tidal-dominated, removing them removes most of the real signal; what is left is the wind-
  and buoyancy-driven part. The amplitude numbers in `[tides]` are the analytic fallback's
  parameters and are unused while the switch is off.
"""

# ╔═╡ 3b5f6e71-60a3-4e4b-9c82-3d2f1b0c6f1a
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

# ╔═╡ 4c607f82-71b4-4e5c-9d93-4e3a2c1d7a2b
# ╠═╡
md"""
## The grid

`nx × ny` over `[domain] lon_min..lon_max`, `lat_min..lat_max`, and `nz = 8` layers placed by
`[grid] depths` — an explicit face list, positive downward, which must span `z_min = -1500`.

Notice what the list is *not*: it is not evenly spaced. Faces cluster near the surface (0, 20,
50, 100, 150 m) where the stratification and the wind mixing act, and stretch to 1500 m below to
cover the column cheaply. That clustering is the decision — an even 8-layer split of 1500 m puts
one cell centre every 187 m and resolves nothing about the thermocline.

`vertical_stretching_mode = "tanh"` here is not used for placing these faces; the explicit list
takes precedence. The `nz_above` / `vertical_break_depth` keys of the `"two_segment"` mode are
unused in `"depths"` mode, and are the switch to flip when a run needs a resolved surface layer
above a resolved deep one.
"""

# ╔═╡ 5d710893-82c5-4e6d-9a04-5f4b3d2e8b3c
# ╠═╡
import GeoOceanModel: run_segment_grid

grid_res = run_segment_grid(; opts = opts)

bg = grid_res.base_grid
println("base grid: Nx ", bg.Nx, "  Ny ", bg.Ny, "  Nz ", bg.Nz)
println("lon span : ", bg.λᶠᵃᵃ[1], "°E to ", bg.λᶠᵃᵃ[end], "°E")
println("faces    : ", opts.vertical_depths)
println("haloes   : Hx ", grid_res.immersed_grid.Hx, "  Hy ", grid_res.immersed_grid.Hy)
println("immersed : ", hasproperty(grid_res.immersed_grid, :immersed_boundary))

# The wet fraction of the sea floor, from the host array the immersion was regridded from. The
# immersed seabed itself is a halo-padded OffsetArray over a CuArray on a GPU run, and neither
# `count` with a predicate nor broadcasting over it is allowed there.
wet = count(x -> x < 0, bathy.elevation)
println("wet seabed: ", wet, " / ", length(bathy.elevation), " cells (",
        round(100 * wet / length(bathy.elevation), digits = 1), "%)")

# ╔═╡ 6e8219a4-93d6-4e7f-9b15-6a5c4e3f9c4d
# ╠═╡
md"""
## The model: the same physics, spelled out

`run_segment_model` assembles a `HydrostaticFreeSurfaceModel` from that grid and the fetched
inputs. This is the slow cell — the first interval compiles the turbulent-closure kernels.

### Parameterizations

- **Equation of state — TEOS-10**, reference density 1025 kg m⁻³ rather than the library's 1020
  default. Buoyancy and N² differ from older linear results: an accuracy change, not a drop-in.
- **Coriolis — spherical**, so the β term survives across the 45.0–46.9 °N domain.
- **Free surface — split-explicit**: barotropic waves subcycled rather than solved implicitly.
- **Vertical turbulence — `turbulence_closure = "nemotke"`**, the prognostic NEMO-TKE scheme: TKE
  is carried as a field and mixing lengths are diagnosed. The cost is one more tracer and a
  tighter step limit; the gain is stratification that responds to the wind.
- **Horizontal diffusion — a fixed `HorizontalScalarDiffusivity(ν = κ = 100.0)`**, hard-coded in
  `build_hydrodynamic_model`. Numerical smear, **not** a calibrated diffusivity, and deliberately
  absent from the TOML so it cannot be mistaken for a tuned parameter.
- **Bottom drag — quadratic** with `cd_drag = 0.0025`, plus a background linear rate
  `bottom_drag = 0.0001` s⁻¹ and a `tke` benthic mixing closure.
- **Surface heat — a constant 50 W m⁻²**, because `bulk_heat_flux = false`. The bulk formulation
  would need the sea-surface temperature, which this Oceananigans version cannot currently give
  a surface boundary condition. The reason is recorded in the TOML.
- **Wind drag — `garratt_1977`** over `u10, v10`.

### Assumptions worth knowing before you read the output

- **Open edges are unconstrained**: the largest simplification in this configuration. Incoming
  water carries no prescribed properties, so a strength of exchange with the open shelf is not
  represented here.
- **Tides are off**: what remains is the wind- and buoyancy-driven part.
- **Initial stratification is analytic** (`hydrography_source = "woa23"` seeds it, and without
  boundary currents the first day is a relaxation toward balance, not an equilibrium).
- **Salinity is not stored.** The output callback writes u, v, w and T, so density read back from
  the store is derived from temperature alone against the reference 35 PSU.
- **`sim_dt = 120` s is only the initial step**: `adaptive_cfl = true` with a 900 s ceiling means
  the step is set dynamically, and the reported step will not be 120 s.
"""

# ╔═╡ 7f9320b5-0a47-4e8a-9c26-7b6d5f4a0d5e
# ╠═╡
import GeoOceanModel: run_segment_model

built = run_segment_model(; opts = opts, immersed_grid = grid_res.immersed_grid)

println("Simpson-Hunter, shallow bank : ", built.simpson_hunter_bank)
println("Simpson-Hunter, deep shelf    : ", built.simpson_hunter_shelf)
built

# ╔═╡ 0a0431c6-1b58-4e9b-9d37-8c7e6f5b1e6f
# ╠═╡
md"""
## Run the simulation, and what it writes

`run_segment_simulation` advances the model and writes the flow into the GeoData store. On the
CLI, `--all` is exactly these cells plus the data and grid segments.

### The output cadence is its own decision

`[storage] output_schedule_seconds = 21600` — one snapshot every 6 hours, independent of
`checkpoint_schedule_seconds`. The two were once the same knob; separating them is the point,
because a 6-hour field cadence over a 3-day run is 13 intervals, while a checkpoint cadence
decides how much work a crash loses. Cadence is a disk and animation choice; checkpoint cadence
is a recovery choice.

### What lands in the store

- **u, v, w, T per interval** — the store keeps the model's native stagger, and the reader blends
  it to cell centres.
- **the grid's coordinates** travel inside the store's attributes, alongside `time.json`, so the
  handover is a directory with nothing beside it.
- **no salinity**, as above.

Re-running with a larger `sim_duration_hours` resumes from the checkpoint, rather than starting
over.
"""

# ╔═╡ 1b1542d7-2c69-4e0c-9e48-9d8f7a6c2f70
# ╠═╡
import GeoOceanModel: run_segment_simulation

sim_res = run_segment_simulation(; opts = opts, model = built.model)

println("flow solution: ", sim_res.output_path)
sim_res

# ╔═╡ 2c2653e8-3d7a-4e1d-9a59-0e9f8b7d3a81
# ╠═╡
md"""
## Read the store back

`extract_hydrodynamic_dataset` takes the run's output **directory** — the thing
`run_segment_simulation` hands back — and reads that store. It is the same path a figure takes,
so what is read here is what gets drawn.

Ask for the **last** interval: the first is the rest state before the wind has acted.

Three checks that mean "this is the real run and not the synthetic fallback":

1. the interval count matches the cadence (`duration / output_schedule_seconds + 1`),
2. `maximum(abs, ds.u)` is non-zero — identically zero over every interval means still water,
3. `T` spans a physical range rather than a constant — a field with no relation to the seabed
   above means the analytic synthesis fired instead of the store.
"""

# ╔═╡ 3d3764f9-4e8b-4e2e-9b6a-1f0a9c8e4b92
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

# ╔═╡ 4e4875a0-5f9c-4e3f-9c7b-2a1b0d9f5c03
# ╠═╡
# Real flow, not zeros and not synthesis.
maximum(abs, ds.u)

# ╔═╡ 5f5986b1-60ad-4e4a-9d8c-3b2c1e0a6d14
# ╠═╡
md"""
## Diagnostics: what the reader derives

`compute_hydrodynamic_diagnostics` takes the four stored fields and derives the rest: density
(degrading to the thermal term when salinity is absent, which here it is), stratification,
Richardson number, vorticity, and the turbulent viscosity and diffusivity from the TKE closure.
Every figure uses it, so a figure's "density" is this number.

A single column is where stratification is legible.
"""

# ╔═╡ 60a697c2-71be-4e5b-9e9d-4c3d2f1b7e25
# ╠═╡
import GeoOceanModel: compute_hydrodynamic_diagnostics
using Statistics

diag = compute_hydrodynamic_diagnostics(ds.lons, ds.lats, ds.depths, ds.u, ds.v, ds.w, ds.temperature, ds.salinity)

imap = argmin(abs.(ds.lons .- mean(ds.lons)))
jmap = argmin(abs.(ds.lats .- mean(ds.lats)))

println("density at the column centre (kg m⁻³), surface → bottom:")
for k in eachindex(ds.depths)
    println(rpad(round(ds.depths[k], digits = 1), 9), " m  →  ", round(diag.density[imap, jmap, k], digits = 3))
end

# ╔═╡ 71b7a8d3-82cf-4e6c-9eae-5d4e3a2c8f36
# ╠═╡
md"""
## Plot a slice

Surface speed and temperature, land masked out. White cells are the emerged land from the seabed
resolved above, which is what makes the coastline visible instead of a blank rectangle.

Cape Breton is heavily fringed by land, so a high dry fraction is expected here; anything
approaching *all* dry would mean the seabed is wrong.

The mask is drawn against **the run's own fetched coastline** — read from
`<output_dir>/inputs/coastline.parquet` — not against a bundled regional outline.
"""

# ╔═╡ 82c8b9e4-93d0-4e7d-9abf-6e5f4b3d9a47
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

# ╔═╡ 93d9c0f5-0a41-4e8e-9bc0-7f6a5c4e0b58
# ╠═╡
md"""
## What to change, and what each change does

Knobs worth turning first, in the order they matter:

- **`[boundaries] ocean_boundary_source`** — set it to `"woa23"` for keyless *constrained* edges
  (temperature and salt, no currents). It is the single upgrade over this config's synthetic
  edges, and it costs nothing but download time.
- **`[tides] enable_tides`** — off gives the wind- and buoyancy-only circulation. Turning the
  analytic tide on (`tides_source = "analytic"`, amplitudes in the section) is keyless and shows
  the machinery; the TPXO9 atlas is the realistic one and is keyless too, but it downloads.
- **`[data] hydrography_source`** — `"woa23"` (keyless) or `"synthetic"` (no download, analytic
  profile, for testing).
- **`[grid] nx / ny / nz`** — the cheapest accuracy-for-cost knob. Doubling `nx` quadruples the
  work; the vertical faces matter more per cell than the horizontal count here.
- **`[grid] depths`** — an explicit face list. Add faces where you want centres.
- **`[storage] output_schedule_seconds`** — 21600 s here is analysis-grade. Drop to 1800 s for an
  animation-grade 145 intervals, at a third more disk.
- **`[hydrodynamics] sim_duration_hours`** — the run length; re-running resumes from the
  checkpoint.
"""

# ╔═╡ 04ea1a06-1b52-4e9f-9cd1-8a7b6d5f1c69
# ╠═╡ Cell order:
# ╠═d7f2c0a4-9e18-4b63-8f52-0a1c5e9b3d64
# ╠═1e0b94f7-6c82-4d51-9b7a-3f8e2c0a6b15
# ╠═2a1c05e8-7d39-4e72-9c8b-4a9f3d1b7c26
# ╠═b3d2e6f9-84a1-4b93-9a0c-5b4f3e2c8d37
# ╠═c4e3f70a-95b2-4ca4-9b1d-6c5f4a3d9e48
# ╠═d5f4081b-0a63-4db5-9c2e-7d6f5b4e0f59
# ╠═e60a192c-1b74-4ec6-9d3f-8e7f6c5f1a6b
# ╠═f71b2a3d-2c8f-4fd7-9e4f-9f8f7d6e2b7c
# ╠═082c3b4e-3d90-4e18-9a5f-0a9f8e7f3c8d
# ╠═193d4c5f-4e01-4f29-9b60-1b0f9f8a4d9e
# ╠═2a4e5d60-5f12-4e3a-9b71-2c1f0a9b5e0f
# ╠═3b5f6e71-60a3-4e4b-9c82-3d2f1b0c6f1a
# ╠═4c607f82-71b4-4e5c-9d93-4e3a2c1d7a2b
# ╠═5d710893-82c5-4e6d-9a04-5f4b3d2e8b3c
# ╠═6e8219a4-93d6-4e7f-9b15-6a5c4e3f9c4d
# ╠═7f9320b5-0a47-4e8a-9c26-7b6d5f4a0d5e
# ╠═0a0431c6-1b58-4e9b-9d37-8c7e6f5b1e6f
# ╠═1b1542d7-2c69-4e0c-9e48-9d8f7a6c2f70
# ╠═2c2653e8-3d7a-4e1d-9a59-0e9f8b7d3a81
# ╠═3d3764f9-4e8b-4e2e-9b6a-1f0a9c8e4b92
# ╠═4e4875a0-5f9c-4e3f-9c7b-2a1b0d9f5c03
# ╠═5f5986b1-60ad-4e4a-9d8c-3b2c1e0a6d14
# ╠═60a697c2-71be-4e5b-9e9d-4c3d2f1b7e25
# ╠═71b7a8d3-82cf-4e6c-9eae-5d4e3a2c8f36
# ╠═82c8b9e4-93d0-4e7d-9abf-6e5f4b3d9a47
# ╠═93d9c0f5-0a41-4e8e-9bc0-7f6a5c4e0b58
# ╠═04ea1a06-1b52-4e9f-9cd1-8a7b6d5f1c69
