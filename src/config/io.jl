"""
    io.jl

Configuration file I/O.

# Reading a TOML file into a dictionary and writing one back. All that lives here is
# files and dictionaries; what the dictionary becomes is in `conversion.jl`.
#
# Every file read here is also checked against `domains.jl`'s section list, because a
# misspelled section is indistinguishable from a missing one and would otherwise let the
# whole of it fall back to defaults.
"""

using TOML


"""
    load_configuration(config_path::AbstractString = find_default_config_path()) -> Dict{String, Any}

Read and parse a centralized `ParticleTracking.toml` file into a nested Julia Dictionary.
A missing or unparseable file is an error, not a silent fall back to defaults.

# Inputs
- `config_path::AbstractString`: Path to the `.toml` (TOML format) file.

# Outputs
- `Dict{String, Any}`: Nested dictionary containing all sectioned parameter settings.
"""
function load_configuration(
    config_path::AbstractString = find_default_config_path()
)::Dict{String, Any}
    # A run must never proceed on defaults it did not ask for: a silently substituted
    # configuration produces output that looks real and is not. `get_default_configuration()`
    # remains available for callers that genuinely want the defaults.
    isempty(config_path) && error(strip(NO_CONFIG_ADVICE, '\n'))
    isfile(config_path) || error(
        "Configuration file not found: $(config_path). Pass --config=<path>."
    )
    try
        parsed = TOML.parsefile(config_path)
        validate_config_sections(parsed, config_path)
        return parsed
    catch err
        error("Failed to parse configuration file at $(config_path): $(err)")
    end
end

"""
    save_configuration(
        config_dict::AbstractDict,
        config_path::AbstractString = "ParticleTracking.toml"
    ) -> String

Serialize a nested configuration dictionary to a centralized `.toml` file in TOML format.

# Inputs
- `config_dict::AbstractDict`: Dictionary of configuration sections and key-values.
- `config_path::AbstractString`: Target destination file path.

# Outputs
- `String`: Path to the written configuration file.
"""
function save_configuration(
    config_dict::AbstractDict,
    config_path::AbstractString = "ParticleTracking.toml"
)::String
    out_dir = dirname(abspath(config_path))
    if !isdir(out_dir)
        mkpath(out_dir)
    end
    open(config_path, "w") do io
        TOML.print(io, config_dict; sorted = true)
    end
    return config_path
end
