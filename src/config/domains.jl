"""
    domains.jl

Which configuration section belongs to which domain, and where each domain's file lives.

The one place that knows this. `conversion.jl` reads a section, and `options.jl` holds a field
for it; before this file the partition existed only implicitly, built by hand in both the
reader and the writer and nowhere else, so nothing could say what a section *meant* or that a
new one had been misfiled.

Three domains, not two. `[domain]`, `[grid]`, `[metadata]`, `[storage]`, `[hardware]`,
`[visualization]` and `[paths]` are read by both the ocean and the biology halves of a run, so
they are not the ocean's to own. They are `:shared`, and they live in the ocean file when the
config is split — the biology of a study is the biology of *this* ocean, and the ocean
declares where it is.
"""

"""
    OCEAN, BIOLOGY, SHARED

The three configuration domains, as `Symbol`s.

`OCEAN` is the physical model: what is forced, what is bounded, what the sea floor is.
`BIOLOGY` is the Lagrangian half: larvae, migration, molting, settlement. `SHARED` is
infrastructure both halves read — the domain and grid, and where output goes.
"""
const OCEAN   = :ocean
const BIOLOGY = :biology
const SHARED  = :shared

"""
    CONFIG_SECTION_DOMAIN

Section name → domain. The single mapping every split and every check reads.
"""
const CONFIG_SECTION_DOMAIN = Dict{String, Symbol}(
    # Shared: read by both halves.
    "metadata"              => SHARED,
    "domain"                => SHARED,
    "grid"                  => SHARED,
    "storage"               => SHARED,
    "runtype"               => SHARED,
    "visualization"         => SHARED,
    # Ocean: what the physical model is forced and bounded by.
    "data"                  => OCEAN,
    "bathymetry"            => OCEAN,
    "atmosphere"            => OCEAN,
    "boundaries"            => OCEAN,
    "tides"                 => OCEAN,
    "bottom_boundary_layer" => OCEAN,
    "hydrodynamics"         => OCEAN,
    # Biology: the Lagrangian half.
    "biology"               => BIOLOGY,
    "dvm"                   => BIOLOGY,
    "molting_and_settlement" => BIOLOGY,
    "tessellation"          => BIOLOGY,
)

"""
    CONFIG_SECTIONS

Every configuration section this package knows, sorted.

An unknown section is an error rather than an ignored one: `conversion.jl` resolves each key
with a default, so a section that is misspelled — `[doman]` instead of `[domain]` — reads as
absent and every parameter inside it silently becomes its default. The run then reports a
domain it was never given, and nothing in the output distinguishes that from having asked for it.
"""
const CONFIG_SECTIONS = sort!(collect(keys(CONFIG_SECTION_DOMAIN)))

"""
    DOMAIN_SECTIONS

Domain → the sections it owns, sorted. Derived from `CONFIG_SECTION_DOMAIN` rather than written
out again, because a second list is a second thing to keep right.
"""
const DOMAIN_SECTIONS = Dict{Symbol, Vector{String}}(
    d => sort!(collect(k for (k, v) in CONFIG_SECTION_DOMAIN if v == d))
    for d in (OCEAN, BIOLOGY, SHARED)
)

"""
    OCEAN_CONFIG, BIOLOGY_CONFIG

Where the two half-configurations live, relative to the workspace.

They are the files that will move into `GeoOceanModel/` and `GeoLarvalTracking/` in phase 6;
until that move they are siblings of this package, and this is the one place their location is
written down.
"""
const OCEAN_CONFIG   = joinpath("GeoOceanModel", "ParticleTracking.toml")
const BIOLOGY_CONFIG = joinpath("GeoLarvalTracking", "ParticleTracking.toml")

"""
    section_domain(section::AbstractString) -> Symbol

Which domain a configuration section belongs to, erroring when the section is not known.

The error names the sections that are known, so a misspelling is corrected from the message
rather than guessed at. That is the point: an unrecognised section would otherwise be ignored,
and a run would proceed on defaults it never chose.
"""
function section_domain(section::AbstractString)::Symbol
    d = get(CONFIG_SECTION_DOMAIN, section, nothing)
    d === nothing && error(
        "Unknown configuration section [$(section)]. Known sections: " *
        "$(join(CONFIG_SECTIONS, ", ")). Check the spelling; a section this package does not " *
        "know is ignored, and every parameter inside it then falls back to its default.")
    return d
end

"""
    validate_config_sections(cfg::AbstractDict, origin::AbstractString = "") -> Nothing

Check every section of a parsed configuration is one this package knows.

Called by `load_configuration`, so the check happens once on the file the run actually named
rather than at each read site. An unrecognised section is an error, not a warning: every
reader skips a key it does not have a default for only if the section is missing, and a
misspelled section looks exactly like a missing one.

`origin` is only used in the message, so the error says which file was wrong.
"""
function validate_config_sections(cfg::AbstractDict, origin::AbstractString = "")::Nothing
    for k in sort!(collect(keys(cfg)))
        k in CONFIG_SECTIONS && continue
        where = isempty(origin) ? "" : " in $(origin)"
        error("Unknown configuration section [$(k)]$(where). Known sections: " *
              "$(join(CONFIG_SECTIONS, ", ")).")
    end
    return nothing
end

"""
    find_domain_config_paths(dir::AbstractString = ".") -> Union{Nothing, NamedTuple}

Locate the two half-configurations under `dir`, or `nothing` when either is absent.

Both must be present: one half alone cannot carry a run, and a layout that sometimes has one
and sometimes two means a run's parameters depend on which files happened to be left on disk.
Returns `(ocean = ..., biology = ...)`.
"""
function find_domain_config_paths(dir::AbstractString = ".")::Union{Nothing, NamedTuple}
    o = joinpath(dir, OCEAN_CONFIG)
    b = joinpath(dir, BIOLOGY_CONFIG)
    (isfile(o) && isfile(b)) || return nothing
    return (ocean = o, biology = b)
end

"""
    merge_domain_configurations(ocean_cfg, biology_cfg;
                                ocean_path = "", biology_path = "") -> Dict{String, Any}

Layer the biology half of a configuration over the ocean half and return one whole config.

The layering is upward, not downward: a section the ocean owns cannot be restated by the
biology half, and a biology section cannot appear in the ocean half at all, because then one
run would carry two values and the winner would be decided by file order rather than by
anything the operator wrote.

Unknown sections are rejected on the way in, through `validate_config_sections`.
"""
function merge_domain_configurations(
    ocean_cfg::AbstractDict,
    biology_cfg::AbstractDict;
    ocean_path::AbstractString = "",
    biology_path::AbstractString = ""
)::Dict{String, Any}
    validate_config_sections(ocean_cfg, ocean_path)
    validate_config_sections(biology_cfg, biology_path)

    ocean_name = isempty(ocean_path) ? "the ocean configuration" : ocean_path
    for section in keys(ocean_cfg)
        d = section_domain(section)
        d === BIOLOGY && error(
            "Section [$(section)] is biology and cannot appear in $(ocean_name). The " *
            "ocean and shared sections belong to the ocean half only; biology's own " *
            "sections are $(join(DOMAIN_SECTIONS[BIOLOGY], ", ")).")
    end

    merged = Dict{String, Any}()
    for (k, v) in ocean_cfg
        merged[k] = v isa AbstractDict ? Dict{String, Any}(v) : v
    end

    bio_name = isempty(biology_path) ? "the biology configuration" : biology_path
    for (section, bio_sec) in biology_cfg
        section_domain(section) === BIOLOGY || error(
            "Section [$(section)] is not biology and cannot appear in $(bio_name). The " *
            "ocean and shared sections belong to the ocean half only.")
        haskey(merged, section) && error(
            "Section [$(section)] appears in both halves. That section is biology's, so it " *
            "belongs only in $(bio_name).")
        merged[section] = bio_sec isa AbstractDict ? Dict{String, Any}(bio_sec) : bio_sec
    end

    return merged
end

"""
    load_domain_configuration(ocean_path, biology_path) -> Dict{String, Any}

Read the two halves of a split configuration and return the merged whole.

Equivalent in content to `load_configuration` on a single complete file. The difference is
that each half is validated as it is read, so a misfiled or misspelled section is reported
here rather than disappearing into a default.
"""
function load_domain_configuration(ocean_path::AbstractString,
                                   biology_path::AbstractString)::Dict{String, Any}
    return merge_domain_configurations(
        load_configuration(ocean_path), load_configuration(biology_path);
        ocean_path = ocean_path, biology_path = biology_path
    )
end
