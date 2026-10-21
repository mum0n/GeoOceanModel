### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 3d1a0c52-7b21-4f6e-9c2b-6f6f4a0a5d01
# ╠═╡
begin
    using Pkg
    using PlutoUI
    using Base64
end


# ╔═╡ c4e8b2a1-9f3d-4c6b-8e15-7d0a2b9f5c18

# ╠═╡

# Activating the package directory from inside the notebook means Pluto's own environment never

# needs this package, and `using GeoOceanModel` sees the working tree. A Pluto save of this file

# stores the activate call, so re-opening it reproduces the same environment.

Pkg.activate(normpath(joinpath(@__DIR__, "..")))

# ╔═╡ 5e2f7d09-4a16-4b83-9c1d-8b3e6f0a1d27

# ╠═╡

using GeoOceanModel

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

# ╔═╡ c2d8f4a1-7b36-4e05-9c8a-1f6b0d3e7a24

# ╠═╡

md"""

## Step 0 — assimilate the run's inputs

Every input this run reads is a *published product*, and each one is assimilated once into the

shared lakehouse rather than being re-downloaded per run. This is the step that happens before

the first segment: the inputs are converted into GeoData stores under

`projects/lakehouse/GeoData/`, and they stay there.

The lakehouse has one directory per half, so the split is visible in the paths themselves:

```

lakehouse/

  GeoData/            shared products, one directory per source

    etopo2022.bathymetry    measured sea floor

    coastline.parquet    Natural Earth coastline, public domain

    era5/    ERA5 surface winds

    tpxo9/    TPXO9-Atlas tidal constituents

    woa23/    World Ocean Atlas temperature and salinity

    glorys/    GLORYS12V1 open-boundary water

  GeoOceanModel/    this half's runs

    scotian_shelf/hydrodynamics.zarr    the flow solution this run writes

```

### What assimilation means here

The register (`GeoDataSources`) holds one description per product: where it is published, which

file it is, what licence it carries, and which tier it belongs to. `fetch_input(:tides)` checks

the store first and downloads only what is not there, so the lakehouse is the cache and the

product list is the truth about what was used.

That is a different kind of assimilation from a conversion step: the inputs are already

GeoData objects (a Zarr array, a Parquet table, a NetCDF field), so nothing is *converted* —

what is converted is the position, from the publisher's URL to a place this machine keeps.

### It is optional

Assimilation is idempotent, and it knows when to stop. A product already in the store is

reused: the run prints `Using existing bathymetry store` and moves on, rather than fetching

1.4 GB again. `force = true` re-fetches when the source is known to have changed.

"""

# ╔═╡ e3f91b52-8c47-4c16-9d0b-2a7c1e4f8b35

# ╠═╡

md"""

### Asking before running

The cell below resolves what this configuration will read and reports which of it is already

on the shelf. Nothing is fetched by asking: it is the difference between "this run needs 4.7 GB

and an account" and discovering it at the boundary segment, after the grid is built.

"""

# ╔═╡ f4a02c63-9d58-4c27-9e1c-3b8d2f5a9c46

# ╠═╡

opts = GeoOceanModel.configuration_to_options(

    GeoOceanModel.load_configuration("configs/scotian_shelf.toml")

)

# ╔═╡ a5b13d74-0e69-4c38-9f2d-4e9c3a6b0d57

# ╠═╡

assimilated = GeoOceanModel.assimilation_status(opts)

# ╔═╡ b6c24e85-1f7a-4d49-9f3e-5f0d4b7c1e68

# ╠═╡

GeoOceanModel.report_assimilation_status(stdout, assimilated)

# ╔═╡ c7d35f96-2a8b-4c5a-9f4f-0b1e5c8d2f79

# ╠═╡

md"""

### What the rows mean

`have` versus `fetch` is this machine's state, not a property of the configuration — the same

config on a fresh install shows almost all of them as `fetch`, and that is the run the lakehouse

exists for. Two facts matter for planning:

- **`boundary` needs an account.** The GLORYS12V1 open-boundary water and initial hydrography

  come from Copernicus Marine, which is free but wants registration; credentials are read from

  `~/.copernicusmarine/credentials`. The run stops without them rather than substituting a

  coarser product, because the open edges would then hold different water and nothing in the

  output would show it.

- **`tides` is the big one.** The TPXO9-Atlas archive is 4.7 GB. It is fetched once and the

  extracted `u_tpxo9.v1.nc` is what every later run reads; the archive itself is not kept.

### Where the run writes

`run_store` is this run's own output, in this half's directory rather than the shared one: the

realized flow solution, read by the larval half through the store's own path. The store carries

the grid, the times and the fields as a GeoData dataset, so a reader needs no sidecar and no

knowledge of how it was written.

"""

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

# ╠═╡  GeoOceanModel/            this package: grid, physics, ocean figures

# ╠═╡    └── configs/*.toml        the shipped run definitions

# ╠═╡  GeoData/                  the abstraction layer: geoload / geosave / landmask

# ╠═╡  GeoDataSources/           the fetchers: what gets downloaded, and where it is cached

# ╠═╡  lakehouse/

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

- `inshore_depth = 20.0` — the depth the landward edge is pinned to (positive downward), so the shoreline is a wet

  cell rather than a cliff.

- `shelf_slope = 180.0` — the total relief / rise across the continental shelf (m).

- `mask_bay_of_fundy = true` — hypertidal shallows set to solid land. That is a deliberate

  exclusion: they generated spurious vertical velocities (~1.3 m/s) and pinned the adaptive

  timestep to ~1 s, and snow crab do not enter the bay.

- `ingest_library = "geodata"` — which library performs the ingestion and regridding.

### `[boundaries]` — the open edges

The domain is a rectangle cut out of a larger sea, so three edges are imaginary lines where water

crosses. `ocean_boundary_source = "glorys12v1"` constrains the temperature, salinity **and**

velocity just outside. WOA23 was the keyless alternative and was set aside for one reason: it

carries no currents, and for a dispersal calculation the volume coming through the edge is the

`sponge_layer_width = 0.45` degrees (~50 km) across `sponge_timescale_seconds = 10800` (3 hours)
— gentle damping over a wider boundary zone placed in the abyssal plain (`lat_min = 41.2°N`,
`lon_max = -56.5°E`), avoiding topographic wave scattering and artificial reflections at the shelf break.

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

## The grid: rotated orthogonal curvilinear shelf grid

`build_shelf_grid` (dispatched through `run_segment_grid`) constructs either a standard geographic spherical grid (`LatitudeLongitudeGrid`) or an orthogonal curvilinear grid (`RotatedLatitudeLongitudeGrid`) rotated to align with the continental margin.

### Curvilinear alignment along the continental margin

- **Intuitive Motivation (Generalists)**: The Scotian Shelf runs diagonally from southwest to northeast (roughly $50^\circ$ clockwise from North). A standard north-south grid simulating this region must include huge areas of deep open ocean in the south and dry mainland in the north that are not relevant to shelf biology. Rotating the coordinate frame parallel to the coastline turns this diagonal strip into a compact rectangular grid, covering the entire habitat corridor from Georges Bank to Cabot Strait with roughly 60% fewer horizontal cells ($14,300$ vs $36,000$).
- **Mathematical Formulation (Specialists)**: An orthogonal curvilinear `RotatedLatitudeLongitudeGrid` is constructed by defining an Euler rotation pole $(\lambda_p, \phi_p)$ via [`compute_rotated_pole`](file:///c:/home/jae/projects/GeoOceanModel/src/data/grid_bathymetry.jl#L142). The local tangent unit vector $\hat{e}_\xi = \sin\theta\,\hat{e}_E + \cos\theta\,\hat{e}_N$ at azimuth $\theta = 50.0^\circ$ and cross-shelf normal $\hat{e}_\eta = \cos\theta\,\hat{e}_E - \sin\theta\,\hat{e}_N$ define the right-handed pole vector $\mathbf{P} = \hat{e}_\xi \times \hat{e}_\eta$. Centered at $(-63.0^\circ\text{E}, 44.0^\circ\text{N})$, the grid spans $L_\xi = 850.0\,\text{km}$ along-shelf and $L_\eta = 220.0\,\text{km}$ cross-shelf across $Nx = 220 \times Ny = 65$ cells.
- **Momentum Advection**: Curvilinear metric variations require `VectorInvariant()` momentum advection to compute vorticity $\boldsymbol{\omega} = \nabla \times \boldsymbol{u}$ on cell vertices without Christoffel symbol singularities, while preserving fifth-order `WENO()` for active temperature and salinity tracers.

### Vertical discretization & standardized depths

- **Intuitive Depth Standardization (Generalists)**: Observers and field biologists describe water depth as a positive distance below the surface ($20\,\text{m}$ inshore, $200\,\text{m}$ at the shelf break). Ocean numerical models measure vertical coordinate $z$ as an elevation with negative numbers below the surface ($z = -20\,\text{m}$, $z = -200\,\text{m}$). Standardizing all user options to positive depths ($D \ge 0$) eliminates human sign mistakes, while the driver automatically handles the negative elevations under the hood.
- **Physical Face Clustering (Specialists)**: The photic mixed layer and thermocline experience rapid wind-driven mixing and air-sea heat fluxes, while deeper waters evolve more gradually. The explicit face list `depths` spans 13 faces down to the shelf floor ($350\,\text{m}$): `[0.0, 15.0, 30.0, 48.0, 70.0, 95.0, 125.0, 160.0, 200.0, 240.0, 275.0, 310.0, 350.0]`. Setting the surface layer to 15 m satisfies the vertical shear Courant condition ($\Delta t \le \Delta z_{\min} / |w|$) under strong wind forcing, enabling larger stable timesteps while resolving the pycnocline.
- **Parameter Definitions**: `vertical_break_depth = 200.0` (m, shelf-slope break), `inshore_depth = 20.0` (m, minimum landward water column floor preventing dried cliffs), `shelf_slope = 180.0` (m, shelf relief scale), and `max_save_depth = 400.0` (m, maximum depth subsampled for output storage).

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

# The grid's coordinate fields are halo-padded OffsetArrays over CuArrays in a GPU run.
# `Array` on such an array fails -- its plain axes do not match the offsets -- and indexing
# one is scalar indexing, which the GPU refuses. Unwrapping the parent, moving it to the
# host, and taking the values works for both a span and a count: neither needs the index
# mapping, only the numbers.
_host_values(a) = Array(parent(a))

# A standard geographic grid carries 1-D longitude at its x-faces; a rotated orthogonal
# shell grid carries geographic longitude as a 2-D cell field instead, so the span has to
# come from the cell centres' extrema.
_lon_host = _host_values(hasproperty(bg, :λᶠᵃᵃ) ? bg.λᶠᵃᵃ : bg.λᶜᶜᵃ)
_lon_span = (Float64(minimum(_lon_host)), Float64(maximum(_lon_host)))
println("lon span : ", _lon_span[1], "°E to ", _lon_span[2], "°E")

println("haloes   : Hx ", grid_res.immersed_grid.Hx, "  Hy ", grid_res.immersed_grid.Hy)

println("immersed : ", hasproperty(grid_res.immersed_grid, :immersed_boundary))

# The wet fraction of the sea floor, from the host array the immersion was regridded from. The

# immersed seabed itself is a halo-padded OffsetArray over a CuArray on a GPU run, and neither

# `count` with a predicate nor broadcasting over it is allowed there.

# `_host_values` (defined above) unwraps the OffsetArray and moves the values to the host. A

# count does not care about the index mapping, only the values.

_elev_host = _host_values(bathy.elevation)

wet = count(x -> x < 0, _elev_host)

println("wet seabed: ", wet, " / ", length(_elev_host), " cells (",

        round(100 * wet / length(_elev_host), digits = 1), "%)")

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

Toggle whether to load pre-computed simulation outputs if present, or force re-running the integration:

"""

# ╔═╡ b0000001-0001-0001-0001-000000000006
# ╠═╡
md"""
## Running the simulation

This notebook **reads** a completed run; it does not integrate the model. The code that does
the integration is shown here for the record, and is run from the command line:

```bash
julia --project=. GeoOceanModelRun.jl --config=$(opts.config_file) --all --overwrite
```

or, equivalently, from inside a Julia session with this package loaded:

```julia
import GeoOceanModel: run_segment_data, run_segment_grid, run_segment_model, run_segment_simulation
opts  = GeoOceanModel.configuration_to_options("$(opts.config_file)")
data  = run_segment_data(; opts = opts)
grid  = run_segment_grid(; opts = opts)
built = run_segment_model(; opts = opts, immersed_grid = grid.immersed_grid)
res   = run_segment_simulation(; opts = opts, model = built.model)
```

`--overwrite` clears a previous store first; without it a re-run resumes from the checkpoint.
The next cell loads whatever the store holds. If there is no store, it stops and says which
command produces one — the decision to run a multi-hour integration belongs to you, not to a
notebook cell.
"""

# ╔═╡ 28937602-3f1b-4c23-9d34-bc60419d2e7a

import GeoOceanModel: output_zarr_path

# The simulation is NOT run here. The code that runs it is shown above, for the record;
# this cell only reads what a previous run left in the store. If the store is absent,
# this cell stops and says so, rather than starting a run that would take hours.

sim_res = begin
    expected_store = output_zarr_path(opts.output_dir)
    store_exists = isdir(expected_store) && !isempty(readdir(expected_store))

    if !store_exists
        error(
            "No hydrodynamic store at $(expected_store).\n" *
            "This notebook reads a completed run; it does not run the model.\n" *
            "To produce one, run either\n" *
            "    julia --project=. GeoOceanModelRun.jl --config=$(opts.config_file) --all --overwrite\n" *
            "from the package root, or uncomment the `run_segment_simulation` line in the cell above.")
    end

    println("Reading hydrodynamic solution from: ", expected_store)
    (output_path = opts.output_dir, model = built.model)
end

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

using GeoData

store_dir = output_zarr_path(sim_res.output_path)

# The store's time axis is a Zarr array, not a side-car JSON file: the GeoData Zarr backend
# keeps the coordinate in the group alongside the field variables.
_times = Float64.(collect(GeoData.geoload(store_dir; backend = :zarr).coords[:time].data))

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

# ╔═╡ 6c57ba46-735f-4e67-9a78-f04853161be0

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

# ╔═╡ 7d68cb57-846f-4e78-9b89-015964272c1f

import GeoOceanModel: compute_hydrodynamic_diagnostics
import GeoOceanModel: animate_hydrodynamic_field

using Statistics

diag = compute_hydrodynamic_diagnostics(ds.lons, ds.lats, ds.depths, ds.u, ds.v, ds.w, ds.temperature, ds.salinity)

imap = argmin(abs.(ds.lons .- mean(ds.lons)))

jmap = argmin(abs.(ds.lats .- mean(ds.lats)))

println("density at the column centre (kg m⁻³), surface → bottom:")

for k in eachindex(ds.depths)

    println(rpad(round(ds.depths[k], digits = 1), 9), " m  →  ", round(diag.density[imap, jmap, k], digits = 3))

end

# ╔═╡ 8e79dc68-9579-4a01-9b90-026a75383c2f

# ╠═╡

md"""
## Interactive Regional Exploration & Model Diagnostics

Explore the hydrodynamics across depths and stations on the Scotian Shelf.
Use the interactive map to select a point on the shelf to evaluate vertical water column stratification,
or adjust the depth slice and colormap controls below.
"""

# ╔═╡ b0000001-0001-0001-0001-000000000001
# ╠═╡
@bind shelf_coords html"""
<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 900px; margin: 0 0 10px 0;">
  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
    <span style="font-size: 13px; font-weight: 600; color: #1e293b;">
      Scotian Shelf & Laurentian Channel Domain (Click on map to probe water column)
    </span>
    <span id="shelf_coord_display" style="font-size: 12px; font-family: monospace; color: #475569; background: #f1f5f9; padding: 2px 8px; border-radius: 4px;">
      Selected: 44.5000°N, -61.0000°W
    </span>
  </div>
  <div id="interactive_ocean_map" style="height: 380px; width: 100%; border-radius: 8px; border: 1px solid #cbd5e1; box-shadow: 0 1px 3px rgba(0,0,0,0.1); z-index: 1;"></div>
</div>

<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

<script>
const container = currentScript.parentElement;
const mapDiv = container.querySelector('#interactive_ocean_map');
const displaySpan = container.querySelector('#shelf_coord_display');

let currentVal = { lon: -61.0, lat: 44.5 };
container.value = currentVal;

function setupOceanMap() {
  if (typeof L === 'undefined') {
    setTimeout(setupOceanMap, 60);
    return;
  }
  if (mapDiv._leaflet_id) {
    return;
  }
  const map = L.map(mapDiv, {
    center: [currentVal.lat, currentVal.lon],
    zoom: 6,
    minZoom: 4,
    maxZoom: 10
  });

  L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}', {
    attribution: 'Tiles &copy; Esri, GEBCO, NOAA',
    maxZoom: 10
  }).addTo(map);

  const shelfBounds = [[42.0, -66.5], [48.5, -57.5]];
  L.rectangle(shelfBounds, {
    color: "#0284c7",
    weight: 1.5,
    fillColor: "#38bdf8",
    fillOpacity: 0.08,
    dashArray: "4, 4"
  }).addTo(map);

  let marker = L.marker([currentVal.lat, currentVal.lon], {
    draggable: true
  }).addTo(map);

  function updatePoint(lat, lon) {
    const rlat = Math.round(lat * 10000) / 10000;
    const rlon = Math.round(lon * 10000) / 10000;
    displaySpan.innerText = `Selected: ${rlat.toFixed(4)}°N, ${rlon.toFixed(4)}°W`;
    container.value = { lon: rlon, lat: rlat };
    container.dispatchEvent(new CustomEvent('input'));
  }

  marker.on('dragend', function(e) {
    const pos = e.target.getLatLng();
    updatePoint(pos.lat, pos.lng);
  });

  map.on('click', function(e) {
    marker.setLatLng(e.latlng);
    updatePoint(e.latlng.lat, e.latlng.lng);
  });

  setTimeout(() => { map.invalidateSize(); }, 200);
}

setupOceanMap();
</script>
"""

# ╔═╡ b0000001-0001-0001-0001-000000000002
# ╠═╡
@bind selected_depth_idx PlutoUI.Slider(1:length(ds.depths), default = argmin(abs.(ds.depths)), show_value = false)

# ╔═╡ 9f80ed79-0a29-4b12-9c01-037b86493093
# ╠═╡
begin
    using CairoMakie

    k_idx = selected_depth_idx
    z_m = round(ds.depths[k_idx], digits = 1)
    
    speed_slice = hypot.(ds.u[:, :, k_idx], ds.v[:, :, k_idx])
    temp_slice = ds.temperature[:, :, k_idx]
    
    fig = Figure(size = (960, 420))
    
    ax_speed = Axis(fig[1, 1], title = "Current Speed |u, v| at $(z_m) m (m/s)", xlabel = "Longitude (°W)", ylabel = "Latitude (°N)")
    hm_speed = heatmap!(ax_speed, ds.lons, ds.lats, speed_slice';
                       colormap = :thermal, nan_color = :white)
    Colorbar(fig[1, 2], hm_speed, label = "m/s")
    
    ax_temp = Axis(fig[1, 3], title = "Temperature T at $(z_m) m (°C)", xlabel = "Longitude (°W)", ylabel = "Latitude (°N)")
    hm_temp = heatmap!(ax_temp, ds.lons, ds.lats, temp_slice';
                      colormap = :thermometer, nan_color = :white)
    Colorbar(fig[1, 4], hm_temp, label = "°C")
    
    fig
end

# ╔═╡ b0000001-0001-0001-0001-000000000003
# ╠═╡
md"""
### Horizontal Circulation & Stratification at Selected Depth
Viewing depth level: **$(round(ds.depths[selected_depth_idx], digits=1)) m** (Surface layer: ~0 m, Deep CIL/slope: < -150 m)
"""

# ╔═╡ b0000001-0001-0001-0001-000000000004
# ╠═╡
md"""
### Vertical Hydrographic Profile at Interactive Probe Station
Column probed at: **$(shelf_coords.lat)°N, $(shelf_coords.lon)°W**
"""

# ╔═╡ b0000001-0001-0001-0001-000000000005
# ╠═╡
begin
    probe_lon = Float64(shelf_coords.lon)
    probe_lat = Float64(shelf_coords.lat)
    
    i_probe = argmin(abs.(ds.lons .- probe_lon))
    j_probe = argmin(abs.(ds.lats .- probe_lat))
    
    col_deps = ds.depths
    col_temp = ds.temperature[i_probe, j_probe, :]
    col_speed = hypot.(ds.u[i_probe, j_probe, :], ds.v[i_probe, j_probe, :])
    col_density = diag.density[i_probe, j_probe, :]
    
    fig_col = Figure(size = (920, 360))
    
    ax_tcol = Axis(fig_col[1, 1], title = "Temperature Profile", xlabel = "Temperature (°C)", ylabel = "Depth (m)")
    lines!(ax_tcol, col_temp, col_deps, color = :firebrick, linewidth = 2.2)
    scatter!(ax_tcol, col_temp, col_deps, color = :firebrick, markersize = 8)
    
    ax_scol = Axis(fig_col[1, 2], title = "Velocity Magnitude", xlabel = "Speed (m/s)", ylabel = "Depth (m)")
    lines!(ax_scol, col_speed, col_deps, color = :dodgerblue, linewidth = 2.2)
    scatter!(ax_scol, col_speed, col_deps, color = :dodgerblue, markersize = 8)
    
    ax_dcol = Axis(fig_col[1, 3], title = "Potential Density", xlabel = "Density (kg/m³)", ylabel = "Depth (m)")
    lines!(ax_dcol, col_density, col_deps, color = :darkgreen, linewidth = 2.2)
    scatter!(ax_dcol, col_density, col_deps, color = :darkgreen, markersize = 8)
    
    fig_col
end


# ╔═╡ a091fe8a-13ab-4c23-9d12-048c97541084

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

# ╔═╡ a1b2c3d4-e5f6-7890-abcd-ef1234567890
# ╠═╡
begin
    using CairoMakie
    import CairoMakie: record
    
    # Animate surface current speed over time from the stored GeoData archive
    anim_path = joinpath(OUTDIR, "outputs", "surface_speed_animation.mp4")
    mkpath(dirname(anim_path))
    
    animate_hydrodynamic_field(
        sim_res.output_path;
        variable = :speed,
        depth_level = 1,
        n_frames = min(12, length(_times)),
        framerate = 2,
        output_path = anim_path,
        title = "Scotian Shelf — Surface Current Speed",
        domain_lon = opts.domain_lon,
        domain_lat = opts.domain_lat,
    )
    
    println("Animation saved to: ", anim_path)
    anim_path
end

# ╔═╡ Cell order:
# ╠═3d1a0c52-7b21-4f6e-9c2b-6f6f4a0a5d01
# ╠═2f9a1e77-5c34-4b8d-9a71-1c2f8b3e5a10
# ╠═1b3c9d84-6a52-4c0e-8f4b-2a7d5e9c0f33
# ╠═a0c1f6b3-2d84-4e79-9b0c-3f5d7a1e6c44
# ╠═c4e8b2a1-9f3d-4c6b-8e15-7d0a2b9f5c18
# ╠═5e2f7d09-4a16-4b83-9c1d-8b3e6f0a1d27
# ╠═c2d8f4a1-7b36-4e05-9c8a-1f6b0d3e7a24
# ╠═e3f91b52-8c47-4c16-9d0b-2a7c1e4f8b35
# ╠═f4a02c63-9d58-4c27-9e1c-3b8d2f5a9c46
# ╠═a5b13d74-0e69-4c38-9f2d-4e9c3a6b0d57
# ╠═b6c24e85-1f7a-4d49-9f3e-5f0d4b7c1e68
# ╠═c7d35f96-2a8b-4c5a-9f4f-0b1e5c8d2f79
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
# ╠═b0000001-0001-0001-0001-000000000006
# ╠═28937602-3f1b-4c23-9d34-bc60419d2e7a
# ╠═39348713-402c-4d34-9e45-cd71520e3f8b
# ╠═4a359824-513d-4e45-9f56-de82631f4a9c
# ╠═5b46a935-624e-4f56-9a67-ef93742050ad
# ╠═6c57ba46-735f-4e67-9a78-f04853161be0
# ╠═7d68cb57-846f-4e78-9b89-015964272c1f
# ╠═8e79dc68-9579-4a01-9b90-026a75383c2f
# ╠═b0000001-0001-0001-0001-000000000001
# ╠═b0000001-0001-0001-0001-000000000002
# ╠═b0000001-0001-0001-0001-000000000003
# ╠═9f80ed79-0a29-4b12-9c01-037b86493093
# ╠═b0000001-0001-0001-0001-000000000004
# ╠═b0000001-0001-0001-0001-000000000005
# ╠═a091fe8a-13ab-4c23-9d12-048c97541084
# ╠═a1b2c3d4-e5f6-7890-abcd-ef1234567890
