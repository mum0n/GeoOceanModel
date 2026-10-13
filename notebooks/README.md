# Notebooks

A Pluto notebook for the ocean model: build the grid, inspect the seabed, run the simulation,
and read the GeoData store back — without re-deriving the pipeline by hand.

| Notebook | Domain | Run length | Inputs |
|---|---|---|---|
| [`scotian_shelf.jl`](scotian_shelf.jl) | the snow-crab Scotian Shelf domain | 72 h at a 1800 s cadence | real, incl. GLORYS12V1 edges (needs a Copernicus account) |
| [`default.jl`](default.jl) | Cape Breton, in the NENE corner | 72 h at a 21600 s cadence | keyless |

Both are **Pluto** notebooks: `.jl` files in Pluto's own format, with `md"""..."""` cells between
the code. They document the run as they perform it — what each option means, which assumptions the
model bakes in, and how each physical process is parameterized.

## Running them

Pluto is deliberately **not** a dependency of this package: a notebook is how you look at the
work, not something the run needs. Launched from the package directory, `Pkg.activate(@__DIR__)`
inside the first cells pins the notebook to this package's environment:

```
julia --project=. -e 'using Pluto; Pluto.run()'
```

then open either file from the notebook list. The first cells also `cd` to the package root, so
the `../lakehouse/...` paths in the configs resolve to `projects/lakehouse/` rather than to a
directory inside the package.

Pluto re-runs a cell and everything downstream of it, so tuning `sim_duration_hours` and watching
the store's interval count move costs no more than the changed part.

## Where the run writes

Each notebook sets `OUTDIR` in its configuration cell: the directory the run writes to — the
fetched bathymetry, coastline and winds, the GeoData store, and the checkpoint. Point it anywhere
to keep a scratch run out of the shared lakehouse:

```julia
const OUTDIR = "D:/scratch/julia/geoceanmodel/runs/scotian_shelf"
```

The notebook re-derives its options from that value, so the TOML itself is never edited.

## What a notebook reads

Every input comes from the shared lakehouse through the GeoData layer:

- bathymetry, coastline and winds via `GeoDataSources`, under `lakehouse/GeoData/`
- the realized flow, written to `lakehouse/GeoOceanModel/<run>/hydrodynamics.zarr`

Cells are ordinary package calls, so each one can be re-run on its own while you inspect what it
returned.
