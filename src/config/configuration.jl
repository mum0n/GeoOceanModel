"""
    configuration.jl

Centralized configuration for `ParticleTracking.jl`.

Assembled from five files, one per kind of thing rather than one per stage:

- `defaults.jl`    paths, study definitions, and every default value
- `domains.jl`     which configuration section belongs to which domain, and where
                   each domain's half-configuration lives
- `options.jl`     the types: runtime options and the three config structs
- `io.jl`          reading and writing TOML
- `conversion.jl`  dictionary <-> options <-> config structs

`domains` is included before `io` because `load_configuration` calls
`validate_config_sections`, and `defaults` is included first because everything falls back on it.
"""

include(joinpath(@__DIR__, "defaults.jl"))
include(joinpath(@__DIR__, "domains.jl"))
include(joinpath(@__DIR__, "options.jl"))
include(joinpath(@__DIR__, "io.jl"))
include(joinpath(@__DIR__, "conversion.jl"))

