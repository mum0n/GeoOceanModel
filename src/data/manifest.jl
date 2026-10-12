"""
    manifest.jl

Provenance registry for every physical input a run consumes.

This module re-exports the provenance framework from `GeoDataSources`. Dataset
declarations, the fetch dispatch table, and the fetchers themselves live there; a
ParticleTracking run adds nothing to that machinery and should not carry a copy of it.
"""
module Manifest

using GeoDataSources.GeoDataManifest

# Re-export the public interface so callers write `Manifest.fetch_input`.
export DATA_SOURCES, DataSource, fetch_input, input_dir, file_digest, data_provenance,
    data_source, describe_data_sources

end # module Manifest
