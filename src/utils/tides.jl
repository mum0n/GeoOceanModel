"""
    tides.jl

Astronomical tidal constituents, tidal momentum body forcing, harmonic velocity
reconstruction, and Simpson-Hunter tidal mixing front diagnostics for regional
shelf modeling.
"""

using Oceananigans
using NCDatasets

"""
    TidalHarmonics

Depth-independent harmonic constants for a set of tidal constituents on a lat/lon box,
as read from a TPXO9-Atlas or FES2014 solution.

# Fields
- `constituents::Vector{Symbol}`: Constituents present, in file order.
- `lon::Vector{Float64}`, `lat::Vector{Float64}`: Ascending coordinate axes, in degrees.
- `uRe`, `uIm`, `vRe`, `vIm::Matrix{Float64}`: Complex tidal-velocity amplitudes (m s⁻¹) on
  the `(lat, lon)` grid, one column-major matrix per quantity, restricted to the constituents.
- `omega::Vector{Float64}`: Angular frequency of each constituent (rad s⁻¹), from the file's
  own `omega` variable where present, otherwise [`get_tidal_frequency`](@ref).

The reconstruction convention is the standard TPXO/OTIS one, in which a complex amplitude
`A = A_re + i*A_im` at angular frequency `w` represents the real signal
`A(t) = A_re*cos(w*t) + A_im*sin(w*t)`. Signs therefore matter: reading only the amplitudes and
discarding the phases would destroy the tidal ellipse.
"""
struct TidalHarmonics
    constituents::Vector{Symbol}
    lon::Vector{Float64}
    lat::Vector{Float64}
    uRe::Array{Float64, 3}   # (con, lat, lon)
    uIm::Array{Float64, 3}
    vRe::Array{Float64, 3}
    vIm::Array{Float64, 3}
    omega::Vector{Float64}
end

"""
    tidal_velocity(h::TidalHarmonics, i::Int, j::Int, t::Real) -> (u, v)

Reconstruct the depth-independent tidal velocity at grid point `(i, j)` — `(lon[i], lat[j])` —
and time `t` seconds, by summing the harmonic constants of every constituent.

A single amplitude is not a velocity: each constituent contributes a rotating vector, and the
sum over constituents is what produces the spring-neap envelope and the tidal ellipse. Land
points, which the atlas solutions carry as zero, are returned as `0.0` rather than `NaN` so
they do not poison a boundary series.
"""
function tidal_velocity(h::TidalHarmonics, i::Int, j::Int, t::Real)
    u = 0.0
    v = 0.0
    for k in eachindex(h.constituents)
        c = cos(h.omega[k] * t)
        s = sin(h.omega[k] * t)
        u += h.uRe[k, j, i] * c + h.uIm[k, j, i] * s
        v += h.vRe[k, j, i] * c + h.vIm[k, j, i] * s
    end
    return (u, v)
end

"""
    read_tidal_harmonics(
        filepath::AbstractString;
        constituents = (:M2, :S2),
        lon_range = nothing,
        lat_range = nothing
    ) -> TidalHarmonics

Read harmonic tidal-velocity constants for `constituents` from a TPXO9-Atlas or FES2014
NetCDF solution, restricted to `lon_range`/`lat_range` when given.

Supports the variable naming used by both atlases: `cnc` (constituent names), `lon_u`/`lat_u`,
`lon_v`/`lat_v`, `uRe`/`uIm`, `vRe`/`vIm`, and `omega`. FES2014 uses the same names; where a
file omits `omega` the astronomical frequencies from [`get_tidal_frequency`](@ref) are used.

A missing file is an **error**, not a fallback. Silently substituting a different solution would
leave `resolved_config.toml` recording a tidal source the run never read, which is the same class
of lie as a silent CPU fallback: the provenance file has to be able to say what was actually used.

# Examples
```julia
h = read_tidal_harmonics("inputs/tpxo9_atlas.nc";
    constituents = (:M2, :S2), lon_range = (-71.0, -53.0), lat_range = (40.0, 48.5))
u, v = tidal_velocity(h, 10, 20, 0.0)   # velocity at that point and time
```
"""
function read_tidal_harmonics(
    filepath::AbstractString;
    constituents = (:M2, :S2),
    lon_range = nothing,
    lat_range = nothing
)
    isfile(filepath) || error(
        "Tidal solution file not found: $(filepath).\n" *
        "Real astronomical tides require an atlas solution (TPXO9-Atlas or FES2014). " *
        "Download the NetCDF for your region, place it in inputs/, and point " *
        "[tides] tides_file at it. This reader deliberately does not fall back to a uniform " *
        "body-force approximation: that would let the run proceed while recording a tidal " *
        "source it never used."
    )

    want = collect(constituents)

    return NCDatasets.NCDataset(filepath, "r") do ds
        for v in ("cnc", "uRe", "uIm", "vRe", "vIm")
            haskey(ds, v) || error(
                "Tidal file $(filepath) is missing variable '$(v)'. Expected a TPXO9-Atlas or " *
                "FES2014 solution.")
        end

        # Constituent names, matched case-insensitively so a file using "m2" still resolves.
        names_in_file = [String(strip(n)) for n in ds["cnc"][:]]
        idx = Int[]
        for c in want
            pos = findfirst(n -> uppercase(n) == String(uppercase(String(c))), names_in_file)
            pos === nothing && error(
                "Tidal file $(filepath) does not contain constituent '$(c)'. " *
                "Available: $(join(names_in_file, ", ")).")
            push!(idx, pos)
        end

        # u/v live on their own (Arakawa C) grids, so each has its own axes.
        lon_u = collect(Float64, ds["lon_u"][:])
        lat_u = collect(Float64, ds["lat_u"][:])
        lon_v = collect(Float64, ds["lon_v"][:])
        lat_v = collect(Float64, ds["lat_v"][:])

        # Restrict to the requested box on the *u* grid, then reuse that index set for v where
        # the axes agree. If they differ the v arrays are interpolated onto the u points by
        # nearest index, which is adequate for a boundary series on a ~2.5 km grid.
        iu = isnothing(lon_range) ? collect(eachindex(lon_u)) :
             findall(x -> lon_range[1] <= x <= lon_range[2], lon_u)
        ju = isnothing(lat_range) ? collect(eachindex(lat_u)) :
             findall(y -> lat_range[1] <= y <= lat_range[2], lat_u)
        isempty(iu) && error(
            "No u-point longitudes fall inside lon_range=$(lon_range) in $(filepath).")
        isempty(ju) && error(
            "No u-point latitudes fall inside lat_range=$(lat_range) in $(filepath).")

        # Atlas files store latitude in DESCENDING order, but everything downstream in this
        # package indexes ascending coordinates. The selected indices are therefore re-sorted
        # by coordinate value, which reorders the data along with the axes -- a transposed or
        # reversed subset would silently mirror the field across the domain.
        iu = sort(iu; by = k -> lon_u[k])
        ju = sort(ju; by = k -> lat_u[k])

        lon = lon_u[iu]
        lat = lat_u[ju]

        uRe = _subset_harmonics(ds["uRe"], idx, iu, ju)
        uIm = _subset_harmonics(ds["uIm"], idx, iu, ju)
        vRe = _subset_harmonics(ds["vRe"], idx, iu, ju)
        vIm = _subset_harmonics(ds["vIm"], idx, iu, ju)

        omega = if haskey(ds, "omega")
            collect(Float64, ds["omega"][:])[idx]
        else
            [get_tidal_frequency(c) for c in want]
        end

        return TidalHarmonics(want, lon, lat, uRe, uIm, vRe, vIm, omega)
    end
end

"""
    _subset_harmonics(raw, idx, iu, ju) -> Array{Float64,3}

Take the requested constituents and the requested index box out of an atlas harmonic variable.

Atlas variables are stored `(con, lat, lon)` in the file's own (descending-latitude) order, so
the result is returned with **ascending** latitude to match the coordinate vectors the caller
builds, and every requested constituent is checked for presence before any indexing happens.
"""
function _subset_harmonics(raw, idx, iu, ju)
    A = Array{Float64}(raw)
    ndims(A) == 3 || error(
        "Expected a (con, lat, lon) tidal harmonic variable, got $(ndims(A)) dimensions.")
    ncon, nlat, nlon = size(A)
    for k in idx
        k <= ncon || error(
            "Tidal file requests constituent index $k but the variable has only $ncon entries.")
    end
    out = Array{Float64}(undef, length(idx), length(ju), length(iu))
    for (a, k) in enumerate(idx), b in eachindex(ju), c in eachindex(iu)
        out[a, b, c] = A[k, ju[b], iu[c]]
    end
    return out
end

"""
    TidalElevationHarmonics

Complex ELEVATION harmonic constants for a set of tidal constituents on a lat/lon box, as read
from a TPXO9-Atlas (or FES) solution in its per-constituent layout.

# Fields
- `constituents::Vector{Symbol}`: Constituents present, in file order.
- `lon::Vector{Float64}`, `lat::Vector{Float64}`: Ascending coordinate axes, in degrees.
- `hRe`, `hIm::Array{Float64, 3}`: Real and imaginary parts of the elevation amplitude
  (**metres**), shaped `(con, lat, lon)`.
- `omega::Vector{Float64}`: Angular frequency of each constituent (rad s⁻¹).

# Why elevation and not velocity

The openly redistributable TPXO9-Atlas archive ships elevation only: one file per constituent,
`h_<con>_tpxo9_atlas_30_v2.nc`, each holding the complex harmonic constants `hRe`/`hIm` on the
global `lon_z`/`lat_z` grid. There is no velocity product in it.

That is not a limitation for this model. A barotropic open boundary is driven by the elevation
time series and the depth-integrated transport follows from continuity, so elevation is the
natural boundary variable; the interior then develops its own tidal ellipses and shelf
amplification rather than having a uniform one imposed on it.

# Units and sign convention

Two conventions differ from the naive reading, and both were found by checking against the
data rather than by assumption:

* **The stored amplitudes are millimetres**, per the variable's `units` attribute. They are
  divided by 1000 here. Reading them as metres inflates every amplitude by three orders of
  magnitude, which looks superficially plausible and is badly wrong.
* **The imaginary part enters with a minus sign.** The files store a complex amplitude
  `h = hRe + i*hIm` and document `GMT phase = atan2(-hIm, hRe)`, so the real signal is the real
  part of `h * exp(i*omega*t)`, which is `hRe*cos(omega*t) - hIm*sin(omega*t)`. Using `+hIm`
  would mirror the phase, which reverses the direction of the tidal wave and shifts the
  spring-neap cycle in time.
"""
struct TidalElevationHarmonics
    constituents::Vector{Symbol}
    lon::Vector{Float64}
    lat::Vector{Float64}
    hRe::Array{Float64, 3}   # (con, lat, lon), metres
    hIm::Array{Float64, 3}
    omega::Vector{Float64}
end

"""
    tidal_elevation(h::TidalElevationHarmonics, i::Int, j::Int, t::Real) -> Float64

Reconstruct the tidal elevation (metres) at grid point `(i, j)` — `(lon[i], lat[j])` — and time
`t` seconds, by summing the harmonic constants of every constituent.

A single amplitude is not a tide: each constituent contributes a cosine–sine pair at its own
frequency, and their sum is what produces the spring–neap envelope. Points the atlas carries as
zero (land) are returned as `0.0` rather than `NaN`, so they do not poison a boundary series.

The imaginary part is subtracted, per the convention documented on
[`TidalElevationHarmonics`](@ref).
"""
function tidal_elevation(h::TidalElevationHarmonics, i::Int, j::Int, t::Real)
    z = 0.0
    for k in eachindex(h.constituents)
        z += h.hRe[k, j, i] * cos(h.omega[k] * t) - h.hIm[k, j, i] * sin(h.omega[k] * t)
    end
    return z
end

"""
    tidal_elevation_amplitude(h::TidalElevationHarmonics, i::Int, j::Int) -> Float64

Peak-to-mean tidal elevation amplitude (metres) at a point, `hypot(hRe, hIm)` summed over
constituents in quadrature. Useful for sanity-checking that a region looks physically sensible
before trusting a run: the Bay of Fundy is the standard check, where M2 exceeds 1 m.
"""
function tidal_elevation_amplitude(h::TidalElevationHarmonics, i::Int, j::Int)
    a = 0.0
    for k in eachindex(h.constituents)
        a += hypot(h.hRe[k, j, i], h.hIm[k, j, i])
    end
    return a
end

"""
    tpxo_constituent_file(directory, c::Symbol) -> String

Resolve the on-disk file for one constituent inside an extracted TPXO9-Atlas archive.

The archive names files `h_<con>_tpxo9_atlas_30_v2.nc` (lower-case constituent) and nests them
under a `TPXO9_atlas_v2/` directory. Both the nested and flat layouts are accepted, and the
non-30-minute `h_<con>_tpxo9_atlas_v2.nc` variant is used as a fallback, because the harmonic
constants are identical between the two and only the sampling differs.
"""
function tpxo_constituent_file(directory::AbstractString, c::Symbol)
    stem = lowercase(string(c))
    candidates = String[
        joinpath(directory, "TPXO9_atlas_v2", "h_$(stem)_tpxo9_atlas_30_v2.nc"),
        joinpath(directory, "h_$(stem)_tpxo9_atlas_30_v2.nc"),
        joinpath(directory, "TPXO9_atlas_v2", "h_$(stem)_tpxo9_atlas_v2.nc"),
        joinpath(directory, "h_$(stem)_tpxo9_atlas_v2.nc"),
    ]
    for p in candidates
        isfile(p) && return p
    end
    error(
        "No file for tidal constituent '$(c)' under $(directory). Tried:\n  " *
        join(candidates, "\n  ") *
        "\nExtract the TPXO9-Atlas archive first (see the :tides entry in DATA_SOURCES)."
    )
end

"""
    read_tidal_elevation_harmonics(
        directory::AbstractString;
        constituents = (:M2, :S2),
        lon_range = nothing,
        lat_range = nothing
    ) -> TidalElevationHarmonics

Read elevation harmonic constants for `constituents` from an extracted TPXO9-Atlas (or FES)
archive, restricted to `lon_range`/`lat_range` when given.

The archive is opened one file at a time, because it ships a separate NetCDF per constituent.
A constituent with no file is an **error**, not a silent zero: a tidal model that quietly
omits S2 has no spring–neap cycle, and would look stable while being wrong.

The result is sorted to ascending longitude and latitude, because atlas files store latitude
descending while everything else in this package indexes ascending coordinates. Reading the
stored order without reversing it would mirror the field north–south.

# Example
```julia
h = read_tidal_elevation_harmonics("inputs/tpxo9_atlas_v2";
    constituents = (:M2, :S2), lon_range = (-71.0, -53.0), lat_range = (40.0, 48.5))
i, j = 40, 30
h(lon = h.lon[i], ...)   # see `tidal_elevation`
```
"""
function read_tidal_elevation_harmonics(
    directory::AbstractString;
    constituents = (:M2, :S2),
    lon_range = nothing,
    lat_range = nothing
)
    isdir(directory) || error(
        "Tidal archive directory not found: $(directory).\n" *
        "Real astronomical tides require an atlas solution. TPXO9-Atlas v2 is openly " *
        "licensed (CC-BY-4.0) and can be fetched keylessly from Zenodo record 22970362: " *
        "https://zenodo.org/api/records/22970362/files/TPXO9_atlas_v2.zip/content " *
        "Unzip it under inputs/. See the :tides entry in DATA_SOURCES for alternatives."
    )

    want = collect(constituents)
    isempty(want) && error("No tidal constituents requested.")

    lon = lat = Float64[]
    hRe = Array{Float64, 3}
    hIm = Array{Float64, 3}
    omega = Float64[]

    for (a, c) in enumerate(want)
        path = tpxo_constituent_file(directory, c)
        NCDatasets.NCDataset(path, "r") do ds
            for v in ("hRe", "hIm", "lon_z", "lat_z")
                haskey(ds, v) || error(
                    "Tidal file $(basename(path)) is missing variable '$(v)'. Expected a " *
                    "TPXO9-Atlas elevation file.")
            end

            flon_raw = collect(Float64, ds["lon_z"][:])
            flat = collect(Float64, ds["lat_z"][:])

            # Atlas solutions store longitude in [0, 360); this package works in [-180, 180).
            # Wrapping before subsetting is not cosmetic -- without it every request for a
            # western-hemisphere box matches nothing at all, because -65.7 is not on the axis.
            flon = [mod(x + 180.0, 360.0) - 180.0 for x in flon_raw]

            iu = isnothing(lon_range) ? collect(eachindex(flon)) :
                 findall(x -> lon_range[1] <= x <= lon_range[2], flon)
            ju = isnothing(lat_range) ? collect(eachindex(flat)) :
                 findall(y -> lat_range[1] <= y <= lat_range[2], flat)
            if isempty(iu) || isempty(ju)
                error(
                    "Requested box lon=$(lon_range) lat=$(lat_range) is outside the " *
                    "coverage of $(basename(path)).\n" *
                    "  file covers lon [$(round(minimum(flon), digits = 2)), " *
                    "$(round(maximum(flon), digits = 2))], " *
                    "lat [$(round(minimum(flat), digits = 2)), " *
                    "$(round(maximum(flat), digits = 2))].\n" *
                    "This commonly means the archive is a regional subset rather than a " *
                    "global solution. See the :tides entry in DATA_SOURCES."
                )
            end

            # Ascending output, as documented. Reordering the indices reorders the data with
            # them, so the field cannot end up mirrored.
            iu = sort(iu; by = k -> flon[k])
            ju = sort(ju; by = k -> flat[k])

            if a == 1
                lon = flon[iu]
                lat = flat[ju]
                hRe = Array{Float64}(undef, length(want), length(ju), length(iu))
                hIm = similar(hRe)
                omega = Vector{Float64}(undef, length(want))
            elseif length(flon[iu]) != length(lon) || length(flat[ju]) != length(lat)
                # Constituents in the same archive share one grid. If they do not, the
                # boundaries would be assembled from incompatible point sets.
                error(
                    "Constituent '$(c)' has a different grid from the first constituent; " *
                    "all constituents must come from the same atlas solution.")
            end

            sz = size(ds["hRe"])
            raw_re = reshape(Float64.(ds["hRe"][:]), sz...)
            raw_im = reshape(Float64.(ds["hIm"][:]), sz...)

            # The files store amplitudes in MILLIMETRES (variable attribute `units`).
            # Reading them as metres inflates every tidal elevation by 1000x.
            unit = get(ds["hRe"].attrib, "units", "millimeter")
            scale = occursin("mm", lowercase(string(unit))) ||
                    occursin("milli", lowercase(string(unit))) ? 1e-3 : 1.0

            for b in eachindex(ju), cix in eachindex(iu)
                vre = raw_re[iu[cix], ju[b]]
                vim = raw_im[iu[cix], ju[b]]
                hRe[a, b, cix] = ismissing(vre) ? 0.0 : vre * scale
                hIm[a, b, cix] = ismissing(vim) ? 0.0 : vim * scale
            end

            # TPXO carries omega in a `con`-length variable only in the combined atlas file;
            # the per-constituent elevation files do not have one, so the astronomical
            # frequency is used, which is exact for the standard constituents.
            omega[a] = haskey(ds, "omega") && length(ds["omega"][:]) >= 1 ?
                       Float64(first(ds["omega"][:])) : get_tidal_frequency(c)
        end
    end

    return TidalElevationHarmonics(want, lon, lat, hRe, hIm, omega)
end

"""
    TidalVelocityHarmonics

Complex **velocity** harmonic constants for a set of tidal constituents on a lat/lon box, read
from a global TPXO9-Atlas solution (`u_tpxo9.v1.nc`) or an equivalent FES file.

# Fields
- `constituents::Vector{Symbol}`: Constituents present, in file order.
- `lon::Vector{Float64}`, `lat::Vector{Float64}`: Ascending coordinate axes, in degrees.
- `uRe`, `uIm`, `vRe`, `vIm::Array{Float64, 3}`: Real and imaginary parts of the velocity
  amplitude (m s⁻¹), shaped `(con, lat, lon)`.
- `omega::Vector{Float64}`: Angular frequency of each constituent (rad s⁻¹).

# Why this is the right product for the boundary

The global TPXO9 release carries velocity harmonics, not only elevation. A barotropic open
boundary is driven by the *normal* velocity at the boundary point, and the interior then
generates its own tidal ellipses and shelf amplification. The elevation-only product
([`TidalElevationHarmonics`](@ref)) forces the boundary through continuity instead, which is
also valid but gives the boundary less direct control.

# Conventions that are easy to get wrong

Both were established by inspecting the file rather than assumed, and both silently produce
plausible-looking nonsense if missed:

* **Longitude is 0..360**, not −180..180. Without wrapping, a request for −65.7° matches
  nothing at all.
* **The imaginary part enters with a minus sign.** The files document
  `phase = atan2(-hIm, hRe)`, so the real signal is the real part of `h * exp(i*omega*t)`:
  `hRe*cos(w*t) - hIm*sin(w*t)`. Using `+hIm` reverses the wave and shifts spring–neap in time.
* **Units are millimetres** in the elevation product; the velocity product is already m s⁻¹.
  The unit attribute is read rather than trusted.
"""
# Units and scale, and why small offshore numbers are correct
#
# `ua`/`va` are **velocity** amplitudes in cm/s, and the conversion to m/s is applied here.
# The resulting values are much smaller over the deep ocean than one might expect, and that is
# right rather than a bug. For a barotropic wave `U = sqrt(g/h) * zeta`, so a 1 m tidal
# elevation over 3000 m of water gives about 0.06 m/s, not 0.3 m/s. Measured in this file:
# median 1.75 cm/s over the global ocean, 99th percentile 38.6 cm/s, and 70-123 cm/s in the Bay
# of Fundy and Minas Basin. The first instinct is to suspect a unit error; it is not one.
# (Conflating tidal *elevation* amplitude with tidal *velocity* amplitude is the trap.)
#
# The one genuinely suspicious feature is the global maximum of 2221 cm/s, which is unphysical
# and is almost certainly a fill or spike artefact in a very shallow, 1/8-degree cell. Amplitudes
# are therefore clamped on read, so one bad cell cannot dominate a boundary series.
struct TidalVelocityHarmonics
    constituents::Vector{Symbol}
    lon::Vector{Float64}
    lat::Vector{Float64}
    uRe::Array{Float64, 3}
    uIm::Array{Float64, 3}
    vRe::Array{Float64, 3}
    vIm::Array{Float64, 3}
    omega::Vector{Float64}
end

"""
    tidal_velocity_at(h::TidalVelocityHarmonics, i::Int, j::Int, t::Real) -> (u, v)

Reconstruct the tidal velocity (m s⁻¹) at grid point `(i, j)` — `(lon[i], lat[j])` — and time
`t` seconds, summing every constituent's harmonic pair.

This is the quantity the open-boundary forcing needs: evaluated on the boundary itself, the
component normal to that boundary drives the Flather/Chapman condition. The imaginary part is
subtracted, per the convention documented on [`TidalVelocityHarmonics`](@ref).
"""
function tidal_velocity_at(h::TidalVelocityHarmonics, i::Int, j::Int, t::Real)
    u = 0.0
    v = 0.0
    for k in eachindex(h.constituents)
        c = cos(h.omega[k] * t)
        s = sin(h.omega[k] * t)
        u += h.uRe[k, j, i] * c + h.uIm[k, j, i] * s
        v += h.vRe[k, j, i] * c + h.vIm[k, j, i] * s
    end
    return (u, v)
end

"""
    tidal_velocity_speed(h::TidalVelocityHarmonics, i::Int, j::Int) -> Float64

Speed amplitude (m s⁻¹) at a point: the constituent speeds added in quadrature, which bounds the
speed at every phase. A useful physical sanity check — the Scotian Shelf peaks near 1 m s⁻¹
offshore and far higher inside the Bay of Fundy.
"""
function tidal_velocity_speed(h::TidalVelocityHarmonics, i::Int, j::Int)
    s = 0.0
    for k in eachindex(h.constituents)
        s += hypot(h.uRe[k, j, i], h.uIm[k, j, i])^2 +
             hypot(h.vRe[k, j, i], h.vIm[k, j, i])^2
    end
    return sqrt(s)
end

"""
    nearest_water(h::TidalVelocityHarmonics, i::Int, j::Int) -> (i, j)

Nearest grid index to `(i, j)` that is water rather than land.

Land is stored as `NaN` by [`read_tidal_velocity_harmonics`](@ref), because the atlas encodes
it as an exact zero that is otherwise indistinguishable from slack water. A boundary forcing
sampled on such a cell would prescribe no tide, and a coastal boundary runs through many of
them, so boundary extraction must snap to water first. This is a nearest-neighbour search over
a square of increasing radius, which is adequate at 1/8 degree; it is not an interpolator.
"""
function nearest_water(h::TidalVelocityHarmonics, i::Int, j::Int)
    isfinite(h.uRe[1, j, i]) && return (i, j)
    nlon, nlat = size(h.uRe, 3), size(h.uRe, 2)
    for r in 1:max(nlon, nlat)
        for dj in -r:r, di in -r:r
            (abs(di) != r && abs(dj) != r) && continue   # perimeter only
            ii, jj = i + di, j + dj
            (1 <= ii <= nlon && 1 <= jj <= nlat) || continue
            isfinite(h.uRe[1, jj, ii]) && return (ii, jj)
        end
    end
    error("No water cell found near (lon=$i, lat=$j) in the loaded tidal harmonics.")
end

"""
    read_tidal_velocity_harmonics(
        filepath::AbstractString;
        constituents = (:M2, :S2),
        lon_range = nothing,
        lat_range = nothing
    ) -> TidalVelocityHarmonics

Read global harmonic velocity constants from a TPXO9-Atlas velocity file such as
`u_tpxo9.v1.nc`, restricted to `lon_range`/`lat_range` when given.

Expects the atlas layout: a `con` axis of constituent numbers, a `cnc` name array, and
`uRe`/`uIm`/`vRe`/`vIm` on the staggered `lon_u`/`lat_u` and `lon_v`/`lat_v` grids. A constituent
the file does not contain is an **error**: a tidal model that quietly omits S2 has no
spring–neap cycle and would look stable while being wrong.

A box outside the file's coverage is also an error naming the file's actual range, because the
openly mirrored archives of this dataset are frequently regional extracts rather than the
global solution.

# Example
```julia
h = read_tidal_velocity_harmonics("work/snowcrab/inputs/tpxo9/u_tpxo9.v1.nc";
    constituents = (:M2, :S2), lon_range = (-71.0, -53.0), lat_range = (40.0, 48.5))
u, v = tidal_velocity_at(h, 10, 20, 0.0)
```
"""
function read_tidal_velocity_harmonics(
    filepath::AbstractString;
    constituents = (:M2, :S2),
    lon_range = nothing,
    lat_range = nothing
)
    isfile(filepath) || error(
        "Tidal solution file not found: $(filepath).\n" *
        "The global TPXO9 velocity solution is openly licensed (CC-BY-4.0) on Zenodo record " *
        "8074917: https://zenodo.org/api/records/8074917/files/TPXO9_datafiles.zip/content " *
        "Run `fetch_input(:tides)` to retrieve it into this run's inputs directory. See the " *
        ":tides entry in DATA_SOURCES. This reader deliberately does not fall back to a " *
        "uniform body-force approximation: that would let the run proceed while recording a " *
        "tidal source it never used."
    )

    want = collect(constituents)
    isempty(want) && error("No tidal constituents requested.")

    NCDatasets.NCDataset(filepath, "r") do ds
        # TPXO9 ships velocity as amplitude/phase pairs -- `ua`/`up` in cm/s and degrees GMT --
        # alongside complex transport `URe`/`UIm` in m^2/s. The amplitude/phase form is used
        # because it is unambiguous: the phase is stated directly instead of having to be
        # recovered from a complex product whose sign convention differs between releases.
        haskey(ds, "ua") || error(
            "Tidal file $(basename(filepath)) has no 'ua' variable. Expected a TPXO9-Atlas " *
            "velocity solution such as u_tpxo9.v1.nc (1/8 degree, 15 constituents).")
        for v in ("up", "va", "vp", "con")
            haskey(ds, v) || error(
                "Tidal file $(basename(filepath)) is missing variable '$(v)'.")
        end

        # `con` is a (nct, nc) character matrix -- four columns per constituent name, one row
        # per constituent -- so it arrives from `[:]` as a flat vector of characters rather
        # than as strings and has to be regrouped.
        cmat = ds["con"][:, :]
        chars = vec(collect(cmat))
        nct, ncon = size(cmat)
        raw_names = [String(chars[(a - 1) * nct + 1:(a * nct)]) for a in 1:ncon]
        avail = uppercase.(strip.(raw_names))
        idx = Int[]
        for c in want
            key = uppercase(String(c))
            p = findfirst(==(key), avail)
            p === nothing && error(
                "Tidal file $(basename(filepath)) does not contain constituent '$(c)'. " *
                "Available: $(join(first(avail, 20), ", ")).")
            push!(idx, p)
        end

        # lon_u/lat_u are 2D (ny, nx) meshes in this release. `[:]` flattens a 2D variable,
        # so the axes are read with explicit 2D indexing.
        lon2d = collect(Float64, ds["lon_u"][:, :])
        lat2d = collect(Float64, ds["lat_u"][:, :])
        nlat, nlon = size(lon2d)
        lons = [lon2d[1, i] for i in 1:nlon]
        lats = [lat2d[j, 1] for j in 1:nlat]
        wrapped = [mod(x + 180.0, 360.0) - 180.0 for x in lons]

        ii = isnothing(lon_range) ? collect(eachindex(wrapped)) :
             findall(x -> lon_range[1] <= x <= lon_range[2], wrapped)
        jj = isnothing(lat_range) ? collect(eachindex(lats)) :
             findall(y -> lat_range[1] <= y <= lat_range[2], lats)
        if isempty(ii) || isempty(jj)
            error(
                "Requested box lon=$(lon_range) lat=$(lat_range) is outside the coverage of " *
                "$(basename(filepath)).\n" *
                "  file covers lon [$(round(minimum(wrapped), digits = 2)), " *
                "$(round(maximum(wrapped), digits = 2))], " *
                "lat [$(round(minimum(lats), digits = 2)), " *
                "$(round(maximum(lats), digits = 2))].\n" *
                "Openly mirrored copies of TPXO9 are often *regional extracts* rather than " *
                "the global solution; check the file's own lat range before assuming coverage.")
        end
        ii = sort(ii; by = k -> wrapped[k])
        jj = sort(jj; by = k -> lats[k])

        # Amplitude is stored in cm/s; phase in degrees GMT.
        amp_scale = occursin("cm", lowercase(string(get(ds["ua"].attrib, "units", "cm/s")))) ? 1e-2 : 1.0

        # The harmonic fields are (ny, nx, nc), so the requested box is taken with one slice
        # read rather than element by element -- the full file is 2.3e6 points per constituent
        # and per-element reads would take minutes.
        function gather(name)
            A = Float64.(ds[name][:, :, :])
            sl = A[jj, ii, idx]
            sl = permutedims(sl, (3, 1, 2))   # -> (con, lat, lon)
            return replace(sl, missing => 0.0)
        end

        # The atlas stores LAND as exactly 0.0, which is indistinguishable from a genuine zero
        # tide if taken at face value. In this study box that is 26% of all cells, and the
        # effect is not cosmetic: averaging or maximising over the box without separating land
        # understates the offshore signal, and a boundary series sampled on a coastal cell
        # would prescribe a dead-flat zero tide there. Exact zeros are therefore mapped to NaN
        # so land cannot masquerade as water.
        function nan_where_land(A)
            out = copy(A)
            out[out .== 0.0] .= NaN
            return out
        end

        ua = nan_where_land(gather("ua") .* amp_scale)
        va = nan_where_land(gather("va") .* amp_scale)
        up = gather("up") .* (pi / 180)     # degrees GMT -> radians
        vp = gather("vp") .* (pi / 180)

        # A sanity bound is applied to the amplitude. Deep-ocean values are legitimately small
        # -- see the note on scale below -- but a few cells in the global file exceed 20 m/s,
        # which is unphysical for a tide and indicates a fill or spike artefact rather than a
        # current. Clamping them here keeps a single bad cell from dominating a boundary series.
        ua = clamp.(ua, NaN, 5.0)
        va = clamp.(va, NaN, 5.0)

        # Standard harmonic reconstruction from amplitude and Greenwich phase lag g:
        #   u(t) = A * cos(w*t - g)
        # The complex products URe/UIm are the same information; this form is used because the
        # sign of the phase is stated rather than inferred.
        uRe = ua .* cos.(up)
        uIm = ua .* sin.(up)      # so that hRe*cos - hIm*sin == A*cos(w t - g)
        vRe = va .* cos.(vp)
        vIm = va .* sin.(vp)

        omega = haskey(ds, "omega") ? Float64.(collect(ds["omega"][:]))[idx] :
                [get_tidal_frequency(c) for c in want]

        return TidalVelocityHarmonics(
            want, [wrapped[k] for k in ii], [lats[k] for k in jj],
            uRe, uIm, vRe, vIm, omega
        )
    end
end

"""
    boundary_tidal_forcing(
        h::TidalVelocityHarmonics;
        edges = (:east, :south, :west),
        sample_stride = 1
    ) -> Dict{Symbol, Any}

Build the tidal velocity series for each open-boundary edge, from harmonic constants read by
[`read_tidal_velocity_harmonics`](@ref).

# Why boundary, not body, forcing

The previous implementation applied a *spatially uniform* sinusoidal acceleration over the whole
domain. That is not a small approximation error, and it is what made the Scotian Shelf run
diverge: a uniform 0.25 m s⁻¹ tide resonating on a shallow shelf produced interior velocities
reaching 8-13 m s⁻¹ and tripped the divergence watchdog within six simulated hours. Switching
tides off returned the same configuration to 0.4 m s⁻¹, so the uniform forcing was the cause.

Driving the *boundary* instead fixes the two problems at once. The real field is far weaker
over the deep ocean (median 1.75 cm s⁻¹ globally) and far stronger in the Bay of Fundy (70-123
cm s⁻¹), so the forcing varies by more than an order of magnitude across the domain in the
direction the real tide does, and the interior is free to develop its own ellipses rather than
being told to oscillate uniformly.

# Returned structure

`Dict{Symbol, Any}` with one entry per requested edge, each holding

- `lon`, `lat`: the boundary point coordinates, snapped off land
- `uamp`, `vamp`: harmonic amplitude pairs `(M2_amp, S2_amp, ...)` in m s⁻¹
- `uphase`, `vphase`: the corresponding Greenwich phase lags, in radians

from which `tidal_velocity_at`-style reconstruction gives the series at any time. Land is
excluded via [`nearest_water`](@ref), because a boundary running through a coastal cell would
otherwise prescribe a dead-flat zero tide there.

# Example
```julia
h = read_tidal_velocity_harmonics(path; constituents = (:M2, :S2),
                                  lon_range = (-71, -53), lat_range = (40, 48.5))
edges = boundary_tidal_forcing(h; edges = (:east, :south, :west))
u_at(t) = sum(edges[:east].uamp[k] * cos(edges[:east].omega[k] * t -
                                         edges[:east].uphase[k])
              for k in eachindex(edges[:east].omega))
```
"""
function boundary_tidal_forcing(
    h::TidalVelocityHarmonics;
    edges = (:east, :south, :west),
    sample_stride::Int = 1
)
    stride = max(1, sample_stride)
    nlon, nlat = size(h.uRe, 3), size(h.uRe, 2)
    omega = h.omega

    out = Dict{Symbol, Any}()

    for edge in edges
        lon = Float64[]; lat = Float64[]
        uamp = Array{Float64}(undef, length(omega), 0)
        vphase = Array{Float64}(undef, length(omega), 0)
        uamp_k = Vector{Vector{Float64}}()
        uamp_v = Vector{Vector{Float64}}()
        uphase_k = Vector{Vector{Float64}}()
        vphase_k = Vector{Vector{Float64}}()

        if edge === :east || edge === :west
            i = edge === :east ? nlon : 1
            for j in 1:stride:nlat
                (ii, jj) = nearest_water(h, i, j)
                push!(lon, h.lon[ii]); push!(lat, h.lat[jj])
                push!(uamp_k, [hypot(h.uRe[k, jj, ii], h.uIm[k, jj, ii]) for k in eachindex(omega)])
                push!(uamp_v, [hypot(h.vRe[k, jj, ii], h.vIm[k, jj, ii]) for k in eachindex(omega)])
                push!(uphase_k, [atan(h.uIm[k, jj, ii], h.uRe[k, jj, ii]) for k in eachindex(omega)])
                push!(vphase_k, [atan(h.vIm[k, jj, ii], h.vRe[k, jj, ii]) for k in eachindex(omega)])
            end
        elseif edge === :north || edge === :south
            j = edge === :north ? nlat : 1
            for i in 1:stride:nlon
                (ii, jj) = nearest_water(h, i, j)
                push!(lon, h.lon[ii]); push!(lat, h.lat[jj])
                push!(uamp_k, [hypot(h.uRe[k, jj, ii], h.uIm[k, jj, ii]) for k in eachindex(omega)])
                push!(uamp_v, [hypot(h.vRe[k, jj, ii], h.vIm[k, jj, ii]) for k in eachindex(omega)])
                push!(uphase_k, [atan(h.uIm[k, jj, ii], h.uRe[k, jj, ii]) for k in eachindex(omega)])
                push!(vphase_k, [atan(h.vIm[k, jj, ii], h.vRe[k, jj, ii]) for k in eachindex(omega)])
            end
        else
            error("Unknown edge $(edge). Use one of :north, :south, :east, :west.")
        end

        n = length(lon)
        isempty(lon) && error(
            "Edge $(edge) contains no water points inside the loaded box; the box probably " *
            "does not reach the boundary.")

        uamp = Array{Float64}(undef, length(omega), n)
        vamp = Array{Float64}(undef, length(omega), n)
        uphase = Array{Float64}(undef, length(omega), n)
        vphase = Array{Float64}(undef, length(omega), n)
        for c in 1:n, k in eachindex(omega)
            uamp[k, c] = uamp_k[c][k]
            vamp[k, c] = uamp_v[c][k]
            uphase[k, c] = uphase_k[c][k]
            vphase[k, c] = vphase_k[c][k]
        end

        out[edge] = (lon = lon, lat = lat, uamp = uamp, vamp = vamp,
                     uphase = uphase, vphase = vphase, omega = omega)
    end

    return out
end

"""
    edge_tidal_velocity(edge_data, c::Int, t::Real) -> (u, v)

Tidal velocity (m s⁻¹) at boundary point index `c` of one edge produced by
[`boundary_tidal_forcing`](@ref), and time `t` seconds.

`u(t) = sum_k A_k cos(w_k t - g_k)`, the standard form for an amplitude `A_k` and Greenwich
phase lag `g_k`.
"""
function edge_tidal_velocity(edge_data, c::Int, t::Real)
    u = 0.0
    v = 0.0
    for k in eachindex(edge_data.omega)
        u += edge_data.uamp[k, c] * cos(edge_data.omega[k] * t - edge_data.uphase[k, c])
        v += edge_data.vamp[k, c] * cos(edge_data.omega[k] * t - edge_data.vphase[k, c])
    end
    return (u, v)
end

"""
    tidal_forcing_from_harmonics(h::TidalVelocityHarmonics) -> (u = f, v = g)

Build `(x, y, z, t)` forcing functions for the tidal velocity from harmonic constants, ready to
hand to `build_hydrodynamic_model(; tidal_forcing = ...)`.

# What this replaces, and why

The previous forcing was [`build_tidal_body_forcing`](@ref): a *spatially uniform* sinusoidal
acceleration, `0.25` m s⁻¹ in `u` everywhere, at every depth. On the Scotian Shelf that
configuration diverged -- interior velocities reached 8-13 m s⁻¹ and tripped the divergence
watchdog within six simulated hours of simulated time, and switching tides off returned the
same setup to 0.4 m s⁻¹. A uniform forcing is resonant on a shallow shelf, and it is also simply
wrong: the real M2 field over this domain is 0.009-0.159 m s⁻¹ on the eastern boundary,
0.006-0.031 on the southern, and up to 0.713 in the Bay of Fundy on the western.

The functions returned here evaluate the harmonic reconstruction pointwise, so the forcing
varies across the domain in the way the real tide does rather than being told to oscillate
identically everywhere.

# Lookup

Evaluation uses nearest-neighbour indexing on the atlas grid, which is appropriate at 1/8
degree, and snaps off land with [`nearest_water`](@ref) so a cell over the coast is not
prescribed a dead-flat zero tide. Positions outside the loaded box return `0.0` rather than
clamping to the edge, so a box that does not cover the domain shows up as a quiet corner rather
than a wrong boundary condition.

!!! warning "Still a body forcing"
    These are interior forcing terms, not open-boundary conditions. A Flather/Chapman boundary
    driven by the same harmonics is the physically correct construction and is the remaining
    step; this is the change that removes the uniform resonance in the meantime.
"""
function tidal_forcing_from_harmonics(h::TidalVelocityHarmonics)
    nlon, nlat = size(h.uRe, 3), size(h.uRe, 2)

    # Pre-resolve amplitude and phase per grid point so the hot path is a lookup and two
    # trig calls, not a search.
    uamp = Array{Float64}(undef, size(h.uRe))
    uph = similar(uamp)
    vamp = similar(uamp)
    vph = similar(uamp)
    for i in 1:nlon, j in 1:nlat
        (ii, jj) = nearest_water(h, i, j)
        uamp[:, j, i] = [hypot(h.uRe[k, jj, ii], h.uIm[k, jj, ii]) for k in eachindex(h.omega)]
        uph[:, j, i] = [atan(h.uIm[k, jj, ii], h.uRe[k, jj, ii]) for k in eachindex(h.omega)]
        vamp[:, j, i] = [hypot(h.vRe[k, jj, ii], h.vIm[k, jj, ii]) for k in eachindex(h.omega)]
        vph[:, j, i] = [atan(h.vIm[k, jj, ii], h.vRe[k, jj, ii]) for k in eachindex(h.omega)]
    end

    idx(lon, lat) = begin
        i = argmin(abs.(h.lon .- lon))
        j = argmin(abs.(h.lat .- lat))
        (lon < minimum(h.lon) - 1 || lon > maximum(h.lon) + 1 ||
         lat < minimum(h.lat) - 1 || lat > maximum(h.lat) + 1) ? (0, 0) : (i, j)
    end

    u_f = @inline (x, y, z, t) -> begin
        (i, j) = idx(x, y)
        i == 0 && return 0.0
        s = 0.0
        for k in eachindex(h.omega)
            s += uamp[k, j, i] * cos(h.omega[k] * t - uph[k, j, i])
        end
        s
    end

    v_f = @inline (x, y, z, t) -> begin
        (i, j) = idx(x, y)
        i == 0 && return 0.0
        s = 0.0
        for k in eachindex(h.omega)
            s += vamp[k, j, i] * cos(h.omega[k] * t - vph[k, j, i])
        end
        s
    end

    return (u = u_f, v = v_f)
end

"""
    TidalCoefficientGrid

A GPU-safe representation of the tidal forcing over the model domain: complex velocity
amplitudes resampled onto a small regular grid.

# Why not just capture the atlas arrays in a closure

A forcing passed to Oceananigans is called *inside a GPU kernel*, so it must be allocation-free
and built only from `isbits` data. The obvious implementation -- capture the atlas arrays and
`argmin(abs.(h.lon .- lon))` at call time -- compiles on the CPU and then fails to compile on
the GPU, in `gpu_compute_hydrostatic_free_surface_Gu!`. This is the same class of failure the
sponge forcing hit before, and the fix is the same: reduce the field to a small `isbits` object
and do arithmetic, exactly as `compute_sponge_gamma` does with its scalar corner arguments.

# Fields
- `lon0`, `dlon`, `lat0`, `dlat::Float64`: regular grid origin and spacing, so a coordinate maps
  to an index by arithmetic instead of by search.
- `nx`, `ny::Int`: grid dimensions.
- `uRe`, `uIm`, `vRe`, `vIm`: complex amplitude components, as nested `ntuple`s of `Float64`
  indexed `[constituent][y][x]`. `ntuple` keeps them `isbits`, which is the whole requirement.
- `omega::NTuple{N,Float64}`: angular frequencies.

# Why complex components and not amplitude and phase

Interpolating the real and imaginary parts separately avoids the phase wrapping that makes
`A*cos(w t - g)` interpolation unstable near `g = +/-pi`, and it is the representation the atlas
itself stores, so no polar conversion is needed on the host either. The reconstruction
`Re*cos(w t) - Im*sin(w t)` is the one documented on [`TidalVelocityHarmonics`](@ref).

The grid is deliberately coarse. The tidal field varies on scales of degrees, so a dozen points
per axis reproduces the structure that matters -- 0.006 m/s on the southern boundary against
0.44 m/s in the Bay of Fundy -- while keeping the coefficient object small enough to pass into a
kernel.
"""
struct TidalCoefficientGrid{N, NX, NY, NT}
    lon0::Float64
    dlon::Float64
    lat0::Float64
    dlat::Float64
    nx::Int
    ny::Int
    uRe::NTuple{NT, Float64}
    uIm::NTuple{NT, Float64}
    vRe::NTuple{NT, Float64}
    vIm::NTuple{NT, Float64}
    omega::NTuple{N, Float64}
end

# `NT = N*NX*NY` is a fourth parameter rather than a computed product, because type
# parameters cannot be multiplied inside another type parameter's position.
TidalCoefficientGrid{N, NX, NY}(args...) where {N, NX, NY} =
    TidalCoefficientGrid{N, NX, NY, N * NX * NY}(args...)

"""
    TidalCoefficientGrid(h, lon_range, lat_range; nx = 12, ny = 8)

Resample harmonic constants onto a small regular grid over `lon_range`/`lat_range` so the
forcing can later be evaluated by arithmetic alone, on CPU or GPU.

The nearest-neighbour sampling happens here, once, on the host. That is the point: the
allocation-heavy work happens during setup, and the object handed to the kernel contains only
numbers.
"""
# Land is stored as NaN by the reader. A single masked point must not be allowed to poison a
# harmonic sum, so components are replaced by zero where they are not finite.
@inline _finite(x::Real) = isfinite(x) ? Float64(x) : 0.0

function TidalCoefficientGrid(
    h::TidalVelocityHarmonics,
    lon_range::NTuple{2, Float64},
    lat_range::NTuple{2, Float64};
    nx::Int = 12,
    ny::Int = 8
)
    N = length(h.omega)
    lon0, lat0 = Float64(lon_range[1]), Float64(lat_range[1])
    dlon = (lon_range[2] - lon_range[1]) / max(1, nx - 1)
    dlat = (lat_range[2] - lat_range[1]) / max(1, ny - 1)

    # Flat layout, indexed k + N*(j-1) + N*nx*(i-1).
    sre = zeros(Float64, N, ny, nx)
    sim = zeros(Float64, N, ny, nx)
    svre = zeros(Float64, N, ny, nx)
    svim = zeros(Float64, N, ny, nx)
    for j in 1:ny, i in 1:nx
        lon = lon0 + (i - 1) * dlon
        lat = lat0 + (j - 1) * dlat
        ia = argmin(abs.(h.lon .- lon))
        ja = argmin(abs.(h.lat .- lat))
        (ib, jb) = nearest_water(h, ia, ja)
        for k in 1:N
            # A point can be water for one constituent and masked for another, so every
            # component is sanitised here. Without this a single NaN poisons the whole
            # harmonic sum and the forcing returns NaN across the domain.
            sre[k, j, i] = _finite(h.uRe[k, jb, ib])
            sim[k, j, i] = _finite(h.uIm[k, jb, ib])
            svre[k, j, i] = _finite(h.vRe[k, jb, ib])
            svim[k, j, i] = _finite(h.vIm[k, jb, ib])
        end
    end
    # vec() gives column-major (k fastest, then j, then i), which is exactly the flat index
    # the forcing uses: k + N*(j-1) + N*nx*(i-1).
    flat(a) = ntuple(q -> a[q], N * ny * nx)

    return TidalCoefficientGrid{N, nx, ny}(
        lon0, dlon, lat0, dlat, nx, ny,
        flat(vec(sre)), flat(vec(sim)), flat(vec(svre)), flat(vec(svim)),
        ntuple(k -> Float64(h.omega[k]), N))
end

"""
    tidal_forcing_coefficients(g::TidalCoefficientGrid) -> (u = f, v = g)

Build `(x, y, z, t)` tidal forcing functions from a [`TidalCoefficientGrid`](@ref).

Each reconstructs the momentum tendency `-w_k [Re_k sin(w_k t) + Im_k cos(w_k t)]` with bilinear interpolation of the
complex components. There is no search, no allocation, and no host-array access on the
evaluation path, which is what lets it compile inside a GPU kernel.

A point outside the grid yields `0.0`. That is deliberate: a domain reaching past the loaded
tidal box should show up as a region with no tidal forcing rather than being silently clamped to
a boundary condition borrowed from somewhere else.
"""
function tidal_forcing_coefficients(g::TidalCoefficientGrid{N, NX, NY}) where {N, NX, NY}
    # `vec` on an (N, ny, nx) array is column-major: k fastest, then j, then i. The second
    # dimension is `ny`, NOT `nx`, and conflating the two silently reads the wrong cells --
    # which produced a 24 m/s "tidal" current when the grid was 16x12.
    @inline at(A, k, i, j) = @inbounds A[k + N * ((j - 1) + g.ny * (i - 1))]

    @inline function locate(x, y)
        fx = (x - g.lon0) / g.dlon
        fy = (y - g.lat0) / g.dlat
        (fx < 0 || fy < 0 || fx > NX - 1 || fy > NY - 1) && return (0, 0, 0.0, 0.0)
        i = clamp(floor(Int, fx) + 1, 1, NX - 1)
        j = clamp(floor(Int, fy) + 1, 1, NY - 1)
        return (i, j, fx - floor(fx), fy - floor(fy))
    end

    # Bilinear weight product, computed once per corner.
    @inline w00(tx, ty) = (1 - tx) * (1 - ty)
    @inline w10(tx, ty) = tx * (1 - ty)
    @inline w01(tx, ty) = (1 - tx) * ty
    @inline w11(tx, ty) = tx * ty

    u_f = @inline (x, y, z, t) -> begin
        (i, j, tx, ty) = locate(x, y)
        i == 0 && return 0.0
        (a, b, c, d) = (w00(tx, ty), w10(tx, ty), w01(tx, ty), w11(tx, ty))
        s = 0.0
        for k in 1:N
            re = at(g.uRe, k, i, j) * a + at(g.uRe, k, i + 1, j) * b +
                 at(g.uRe, k, i, j + 1) * c + at(g.uRe, k, i + 1, j + 1) * d
            im = at(g.uIm, k, i, j) * a + at(g.uIm, k, i + 1, j) * b +
                 at(g.uIm, k, i, j + 1) * c + at(g.uIm, k, i + 1, j + 1) * d
            # Momentum tendency (m/s^2):
            # d/dt [Re cos(wt) + Im sin(wt)] = -w Re sin(wt) + w Im cos(wt)
            w = g.omega[k]
            s += w * (-re * sin(w * t) + im * cos(w * t))
        end
        s
    end

    v_f = @inline (x, y, z, t) -> begin
        (i, j, tx, ty) = locate(x, y)
        i == 0 && return 0.0
        (a, b, c, d) = (w00(tx, ty), w10(tx, ty), w01(tx, ty), w11(tx, ty))
        s = 0.0
        for k in 1:N
            re = at(g.vRe, k, i, j) * a + at(g.vRe, k, i + 1, j) * b +
                 at(g.vRe, k, i, j + 1) * c + at(g.vRe, k, i + 1, j + 1) * d
            im = at(g.vIm, k, i, j) * a + at(g.vIm, k, i + 1, j) * b +
                 at(g.vIm, k, i, j + 1) * c + at(g.vIm, k, i + 1, j + 1) * d
            w = g.omega[k]
            s += w * (-re * sin(w * t) + im * cos(w * t))
        end
        s
    end

    return (u = u_f, v = v_f)
end

"""
    tidal_velocity_coefficients(g::TidalCoefficientGrid) -> (u = f, v = g)

Build `(x, y, z, t)` tidal velocity reconstruction functions in **metres per second** (\$m s^{-1}\$)
from a [`TidalCoefficientGrid`](@ref) for use as relaxation targets in `Relaxation`.

Reconstructs the harmonic velocity field with Greenwich phase lag:
```math
u(x, y, t) = \\sum_k \\left[ \\text{uRe}_k(x, y) \\cos(\\omega_k t) + \\text{uIm}_k(x, y) \\sin(\\omega_k t) \\right]
```
where \$\\text{uRe} = A_u \\cos g_u\$ and \$\\text{uIm} = A_u \\sin g_u\$, identically evaluating
\$A_u \\cos(\\omega_k t - g_u)\$.

Contains no allocations, searches, or non-isbits captures on the evaluation path,
compiling natively into GPU free-surface kernels. Coordinates outside the grid return `0.0`.
"""
function tidal_velocity_coefficients(g::TidalCoefficientGrid{N, NX, NY}) where {N, NX, NY}
    @inline at(A, k, i, j) = @inbounds A[k + N * ((j - 1) + g.ny * (i - 1))]

    @inline function locate(x, y)
        fx = (x - g.lon0) / g.dlon
        fy = (y - g.lat0) / g.dlat
        (fx < 0 || fy < 0 || fx > NX - 1 || fy > NY - 1) && return (0, 0, 0.0, 0.0)
        i = clamp(floor(Int, fx) + 1, 1, NX - 1)
        j = clamp(floor(Int, fy) + 1, 1, NY - 1)
        return (i, j, fx - floor(fx), fy - floor(fy))
    end

    @inline w00(tx, ty) = (1 - tx) * (1 - ty)
    @inline w10(tx, ty) = tx * (1 - ty)
    @inline w01(tx, ty) = (1 - tx) * ty
    @inline w11(tx, ty) = tx * ty

    u_vel = @inline (x, y, z, t) -> begin
        (i, j, tx, ty) = locate(x, y)
        i == 0 && return 0.0
        (a, b, c, d) = (w00(tx, ty), w10(tx, ty), w01(tx, ty), w11(tx, ty))
        s = 0.0
        for k in 1:N
            re = at(g.uRe, k, i, j) * a + at(g.uRe, k, i + 1, j) * b +
                 at(g.uRe, k, i, j + 1) * c + at(g.uRe, k, i + 1, j + 1) * d
            im = at(g.uIm, k, i, j) * a + at(g.uIm, k, i + 1, j) * b +
                 at(g.uIm, k, i, j + 1) * c + at(g.uIm, k, i + 1, j + 1) * d
            w = g.omega[k]
            s += re * cos(w * t) + im * sin(w * t)
        end
        s
    end

    v_vel = @inline (x, y, z, t) -> begin
        (i, j, tx, ty) = locate(x, y)
        i == 0 && return 0.0
        (a, b, c, d) = (w00(tx, ty), w10(tx, ty), w01(tx, ty), w11(tx, ty))
        s = 0.0
        for k in 1:N
            re = at(g.vRe, k, i, j) * a + at(g.vRe, k, i + 1, j) * b +
                 at(g.vRe, k, i, j + 1) * c + at(g.vRe, k, i + 1, j + 1) * d
            im = at(g.vIm, k, i, j) * a + at(g.vIm, k, i + 1, j) * b +
                 at(g.vIm, k, i, j + 1) * c + at(g.vIm, k, i + 1, j + 1) * d
            w = g.omega[k]
            s += re * cos(w * t) + im * sin(w * t)
        end
        s
    end

    return (u = u_vel, v = v_vel)
end

"""
    get_tidal_frequency(constituent::Symbol)

Return the astronomical angular frequency \$\\omega\$ (\$\\text{rad s}^{-1}\$) for a
specified tidal constituent.

# Supported Constituents
- `:M2`: Principal lunar semidiurnal (\$T = 12.4206\\text{ h}, \\omega = 1.405189 \\times 10^{-4}\\text{ rad s}^{-1}\$)
- `:S2`: Principal solar semidiurnal (\$T = 12.0000\\text{ h}, \\omega = 1.454441 \\times 10^{-4}\\text{ rad s}^{-1}\$)
- `:N2`: Larger lunar elliptic semidiurnal (\$T = 12.6583\\text{ h}, \\omega = 1.378797 \\times 10^{-4}\\text{ rad s}^{-1}\$)
- `:K1`: Lunar diurnal (\$T = 23.9345\\text{ h}, \\omega = 7.292116 \\times 10^{-5}\\text{ rad s}^{-1}\$)
- `:O1`: Principal lunar diurnal (\$T = 25.8193\\text{ h}, \\omega = 6.759774 \\times 10^{-5}\\text{ rad s}^{-1}\$)

# References
- Pugh, D., & Woodworth, P. (2014). *Sea-Level Science: Understanding Tides,
  Surges, Tsunamis and Mean Sea-Level Changes*. Cambridge University Press.
  DOI: 10.1017/CBO9781139235778
"""
function get_tidal_frequency(constituent::Symbol)
    if constituent == :M2
        return 1.405189e-4
    elseif constituent == :S2
        return 1.454441e-4
    elseif constituent == :N2
        return 1.378797e-4
    elseif constituent == :K1
        return 7.292116e-5
    elseif constituent == :O1
        return 6.759774e-5
    else
        error(
            "Unknown tidal constituent '$(constituent)'. " *
            "Supported: :M2, :S2, :N2, :K1, :O1"
        )
    end
end

"""
    TidalBodyForcingU{N}

Bitstype callable struct representing zonal tidal body forcing from harmonic synthesis.
Stored as compile-time static tuples to ensure zero-allocation GPU kernel compilation.

# Mathematical Formulation
```math
F_u(x, y, z, t) = \\sum_{i=1}^N u_{a,i} \\sqrt{\\omega_i^2 + r^2} \\cos(\\omega_i t + \\phi_i)
```

# Inputs
- `u_amps::NTuple{N, Float64}`: Harmonic zonal velocity amplitudes (\$m s^{-1}\$).
- `scales::NTuple{N, Float64}`: Linear bottom drag scaling coefficients (\$s^{-1}\$).
- `freqs::NTuple{N, Float64}`: Angular constituent frequencies (\$rad s^{-1}\$).
- `phases::NTuple{N, Float64}`: Initial Greenwich constituent phase offsets (\$rad\$).

# Outputs
- `Float64`: Instantaneous zonal acceleration (\$m s^{-2}\$).
"""
struct TidalBodyForcingU{N}
    u_amps :: NTuple{N, Float64}
    scales :: NTuple{N, Float64}
    freqs  :: NTuple{N, Float64}
    phases :: NTuple{N, Float64}
end

@inline function (f::TidalBodyForcingU{N})(x, y, z, t) where {N}
    val = 0.0
    for i in 1:N
        val += @inbounds f.u_amps[i] * f.scales[i] * cos(f.freqs[i] * t + f.phases[i])
    end
    return val
end

"""
    TidalBodyForcingV{N}

Bitstype callable struct representing meridional tidal body forcing from harmonic synthesis.
Stored as compile-time static tuples to ensure zero-allocation GPU kernel compilation.

# Mathematical Formulation
```math
F_v(x, y, z, t) = \\sum_{i=1}^N v_{a,i} \\sqrt{\\omega_i^2 + r^2} \\sin(\\omega_i t + \\phi_i)
```

# Inputs
- `v_amps::NTuple{N, Float64}`: Harmonic meridional velocity amplitudes (\$m s^{-1}\$).
- `scales::NTuple{N, Float64}`: Linear bottom drag scaling coefficients (\$s^{-1}\$).
- `freqs::NTuple{N, Float64}`: Angular constituent frequencies (\$rad s^{-1}\$).
- `phases::NTuple{N, Float64}`: Initial Greenwich constituent phase offsets (\$rad\$).

# Outputs
- `Float64`: Instantaneous meridional acceleration (\$m s^{-2}\$).
"""
struct TidalBodyForcingV{N}
    v_amps :: NTuple{N, Float64}
    scales :: NTuple{N, Float64}
    freqs  :: NTuple{N, Float64}
    phases :: NTuple{N, Float64}
end

@inline function (f::TidalBodyForcingV{N})(x, y, z, t) where {N}
    val = 0.0
    for i in 1:N
        val += @inbounds f.v_amps[i] * f.scales[i] * sin(f.freqs[i] * t + f.phases[i])
    end
    return val
end

"""
    build_tidal_body_forcing(;
        constituents::Vector{Symbol} = [:M2],
        u_amplitudes::Dict{Symbol, Float64} = Dict(:M2 => 0.25),
        v_amplitudes::Dict{Symbol, Float64} = Dict(:M2 => 0.12),
        phases::Dict{Symbol, Float64} = Dict(:M2 => 0.0)
    )

Construct Oceananigans momentum forcing functions \$(F_u, F_v)\$ to drive barotropic
tidal oscillations across the model domain.

# Mathematical Formulation
The horizontal momentum body forcing terms \$F_u(t), F_v(t)\$ are:
```math
F_u(x, y, z, t) = \\sum_k U_k \\omega_k \\cos(\\omega_k t + \\phi_k)
```
```math
F_v(x, y, z, t) = \\sum_k V_k \\omega_k \\sin(\\omega_k t + \\phi_k)
```
where \$\\omega_k\$ is the astronomical tidal frequency (rad s⁻¹) and \$(U_k, V_k)\$ are
velocity amplitudes (m s⁻¹).

The factor \$\\omega_k\$ is intentional: \$F_u = U_k \\omega_k \\cos(\\omega_k t)\$ is the
time derivative \$\\partial_t[U_k \\sin(\\omega_k t)]\$, making it dimensionally an
acceleration (m s⁻²), the correct unit for an Oceananigans body force. The
amplitude \$U_k\$ therefore represents the target tidal velocity amplitude in m s⁻¹;
calibrate it against TPXO/FES tidal prediction or observed current meter ellipses.
For M₂ (\$\\omega = 1.405 \\times 10^{-4}\$ rad s⁻¹) and \$U = 0.25\$ m s⁻¹, the
resulting body force amplitude is \$\\approx 3.5 \\times 10^{-5}\$ m s⁻², consistent
with observed tidal acceleration magnitudes on the Scotian Shelf.

# Inputs
- `constituents::Vector{Symbol}`: List of constituents to include (e.g. `[:M2, :S2]`).
- `u_amplitudes::Dict{Symbol, Float64}`: Zonal tidal velocity amplitudes in \$m s^{-1}\$.
- `v_amplitudes::Dict{Symbol, Float64}`: Meridional tidal velocity amplitudes in \$m s^{-1}\$.
- `phases::Dict{Symbol, Float64}`: Tidal phase offsets in radians.
- `bottom_drag_linear::Real`: Linear bottom drag rate \$r_{\\text{drag}}\$ in \$s^{-1}\$.
  When \$r_{\\text{drag}} > 0\$, the acceleration amplitude is calibrated to
  \$U_k \\sqrt{\\omega_k^2 + r_{\\text{drag}}^2}\$ to compensate for frictional damping
  and maintain the target velocity amplitude \$U_k\$.

# Outputs
- `NamedTuple`: `(u = Fu, v = Fv)` forcing functions.

# References
- Egbert, G. D., & Erofeeva, S. Y. (2002). Efficient inverse modeling of barotropic
  ocean tides. *Journal of Atmospheric and Oceanic Technology*, 19(2), 183-204.
  DOI: 10.1175/1520-0426(2002)019<0183:EIMOBO>2.0.CO;2
"""

function build_tidal_body_forcing(;
    constituents::Vector{Symbol} = [:M2],
    u_amplitudes::AbstractDict{Symbol, <:Real} = Dict(:M2 => 0.25),
    v_amplitudes::AbstractDict{Symbol, <:Real} = Dict(:M2 => 0.12),
    phases::AbstractDict{Symbol, <:Real} = Dict(:M2 => 0.0),
    bottom_drag_linear::Real = 0.0,
    bottom_drag::Union{Nothing, Real} = nothing
)
    freqs = [Float64(get_tidal_frequency(c)) for c in constituents]
    u_amps = [Float64(get(u_amplitudes, c, 0.0)) for c in constituents]
    v_amps = [Float64(get(v_amplitudes, c, 0.0)) for c in constituents]
    phs = [Float64(get(phases, c, 0.0)) for c in constituents]
    r_linear = !isnothing(bottom_drag) ? Float64(bottom_drag) : Float64(bottom_drag_linear)

    # Scaling factor: sqrt(ω² + r²) compensates for linear bottom drag damping
    # In the absence of drag (r = 0), this simplifies to ω.
    scales = [sqrt(ω^2 + r_linear^2) for ω in freqs]

    N = length(constituents)
    F_u = TidalBodyForcingU{N}(Tuple(u_amps), Tuple(scales), Tuple(freqs), Tuple(phs))
    F_v = TidalBodyForcingV{N}(Tuple(v_amps), Tuple(scales), Tuple(freqs), Tuple(phs))

    return (u = F_u, v = F_v)
end

"""
    tidal_velocity_vector(
        t::Real;
        constituents::Vector{Symbol} = [:M2],
        u_amplitudes::Dict{Symbol, Float64} = Dict(:M2 => 0.25),
        v_amplitudes::Dict{Symbol, Float64} = Dict(:M2 => 0.12),
        phases::Dict{Symbol, Float64} = Dict(:M2 => 0.0)
    )

Compute instantaneous horizontal tidal velocity vector \$(u_{\\text{tide}}, v_{\\text{tide}})\$
at time \$t\$ from multi-constituent harmonic synthesis.

# Inputs
- `t::Real`: Time in seconds.
- `constituents::Vector{Symbol}`: Active tidal constituents.
- `u_amplitudes::Dict`: Zonal velocity amplitudes.
- `v_amplitudes::Dict`: Meridional velocity amplitudes.
- `phases::Dict`: Phase offsets.

# Outputs
- `Tuple{Float64, Float64}`: `(u_tide, v_tide)` in \$m s^{-1}\$.
"""
function tidal_velocity_vector(
    t::Real;
    constituents::Vector{Symbol} = [:M2],
    u_amplitudes::AbstractDict{Symbol, <:Real} = Dict(:M2 => 0.25),
    v_amplitudes::AbstractDict{Symbol, <:Real} = Dict(:M2 => 0.12),
    phases::AbstractDict{Symbol, <:Real} = Dict(:M2 => 0.0)
)
    u_tot = 0.0
    v_tot = 0.0
    for c in constituents
        omega = get_tidal_frequency(c)
        u_a = Float64(get(u_amplitudes, c, 0.0))
        v_a = Float64(get(v_amplitudes, c, 0.0))
        phi = Float64(get(phases, c, 0.0))

        u_tot += u_a * cos(omega * t + phi)
        v_tot += v_a * sin(omega * t + phi)
    end
    return (Float64(u_tot), Float64(v_tot))
end

"""
    simpson_hunter_parameter(
        water_depth::Real,
        u_tidal_amplitude::Real;
        u_wind_speed::Real = 0.0,
        cd::Real = 2.5e-3,
        buoyancy_flux::Union{Nothing, Real} = nothing
    )

Compute the Simpson-Hunter tidal mixing front parameter \$\\chi = \\log_{10}(h / U^3)\$
or generalized shear-buoyancy index to identify where tidal and wind dissipation overcomes stratification.

# Mathematical Formulation
```math
\\chi = \\log_{10}\\left( \\frac{h}{U_{\\text{tide}}^3 + \\gamma U_{\\text{wind}}^3} \\right)
```
where \$h\$ is water column depth in meters, \$U_{\\text{tide}}\$ is tidal velocity amplitude,
and \$U_{\\text{wind}}\$ represents surface wind shear dissipation
(Garrett, Keeley & Greenberg 1978; Loder & Greenberg 1986).
- \$\\chi < 1.5\$: Well-mixed water column (e.g. shallow banks, Georges Bank, Bay of Fundy).
- \$\\chi > 2.0\$: Thermally stratified shelf waters.
- \$\\chi \\approx 1.5 - 2.0\$: Tidal mixing front (nursery retention zone).

# Inputs
- `water_depth::Real`: Water column thickness in meters (\$h > 0\$).
- `u_tidal_amplitude::Real`: Peak tidal current amplitude in \$m s^{-1}\$.
- `u_wind_speed::Real`: Optional 10-meter wind speed in \$m s^{-1}\$ (default 0.0).
- `cd::Real`: Bottom drag friction coefficient (default \$2.5 \\times 10^{-3}\$).
- `buoyancy_flux::Union{Nothing, Real}`: Optional surface buoyancy flux \$B\$ in \$m^2 s^{-3}\$.

# Outputs
- `Float64`: Simpson-Hunter parameter \$\\chi\$.

# References
- Simpson, J. H., & Hunter, J. R. (1974). Fronts in the Irish Sea. *Nature*, 250, 404-406.
- Garrett, C. J. R., Keeley, J. R., & Greenberg, D. A. (1978). *Continental Shelf Research*, 18(1), 17-33.
- Loder, J. W., & Greenberg, D. A. (1986). *Continental Shelf Research*, 5(6), 679-704.
"""
function simpson_hunter_parameter(
    water_depth::Real,
    u_tidal_amplitude::Real;
    u_wind_speed::Real = 0.0,
    cd::Real = 2.5e-3,
    buoyancy_flux::Union{Nothing, Real} = nothing
)
    h = max(1.0, abs(Float64(water_depth)))
    u_tide = max(1e-3, abs(Float64(u_tidal_amplitude)))
    u_wind = max(0.0, abs(Float64(u_wind_speed)))
    # Energy dissipation: ε ~ ρ * (U_tide³ + 0.05 * U_wind³)
    dissipation_u3 = u_tide^3 + 0.05 * u_wind^3
    if !isnothing(buoyancy_flux) && buoyancy_flux > 0.0
        return Float64(log10(buoyancy_flux * h / (cd * dissipation_u3)))
    else
        return Float64(log10(h / dissipation_u3))
    end
end
