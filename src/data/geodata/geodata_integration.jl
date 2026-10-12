"""
    geodata_integration.jl

Oceananigans-dependent GeoData functions for ParticleTracking.

Only functions requiring Oceananigans grid types are here.
All other GeoDataSources functions should be called directly from 
"""
module GeoDataIntegration

using GeoData
using GeoDataSources
using GeoData.GeoDataTypes: GeoDataset, GeoArray, Dimension, CoordinateSystem
using Oceananigans
using Oceananigans.Grids: LatitudeLongitudeGrid
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid

# Export only Oceananigans-dependent functions
export
    georegrid_to_model_grid,
    build_immersed_grid_from_geodata

"""
    georegrid_to_model_grid(src_ds::GeoDataset, model_grid; vars=String[]) -> GeoDataset

Regrid a source GeoDataset to match an Oceananigans model grid.
"""
function georegrid_to_model_grid(src_ds::GeoDataset, model_grid; vars::Vector{String} = String[])
    base_g = model_grid isa ImmersedBoundaryGrid ? model_grid.underlying_grid : model_grid
    
    target_lons = collect(Float64, base_g.λᶜᵃᵃ)[base_g.Hx+1 : base_g.Hx+base_g.Nx]
    target_lats = collect(Float64, base_g.φᵃᶜᵃ)[base_g.Hy+1 : base_g.Hy+base_g.Ny]
    
    isempty(vars) && (vars = collect(keys(src_ds.variables)))
    
    # Build target GeoDataset
    lon_dim = Dimension(name=:lon, size=length(target_lons), coords=target_lons, units="degrees_east")
    lat_dim = Dimension(name=:lat, size=length(target_lats), coords=target_lats, units="degrees_north")
    
    target_coords = Dict(
        :lon => GeoArray(target_lons, (lon_dim,), CoordinateSystem(crs="EPSG:4326"), Dict("units" => "degrees_east")),
        :lat => GeoArray(target_lats, (lat_dim,), CoordinateSystem(crs="EPSG:4326"), Dict("units" => "degrees_north"))
    )
    target_dims = Dict(:lon => lon_dim, :lat => lat_dim)
    
    # Add depth if source has it
      # Depth is optional, but which name it travels under is not fixed. Resolve it
      # through `axis`, which consults the alias table; an absent axis raises, and that
      # is the signal to skip the dimension rather than a silent miss on a spelling
      # this function did not anticipate.
      src_deps = try
          first(GeoData.axis(src_ds, :depth))
      catch
          nothing
      end
      if src_deps !== nothing
          dep_dim = Dimension(name=:depth, size=length(src_deps), coords=src_deps, units="meters")
          target_coords[:depth] = GeoArray(src_deps, (dep_dim,), CoordinateSystem(crs="EPSG:4326"), Dict("units" => "meters"))
          target_dims[:depth] = dep_dim
      end
    
    target_ds = GeoDataset(Dict{String, GeoArray}(), target_coords, target_dims, 
                          CoordinateSystem(crs="EPSG:4326"), Dict{String, Any}(), nothing, "target_grid")
    
    return GeoData.georegrid(src_ds, target_ds; vars = vars)
end

"""
    build_immersed_grid_from_geodata(grid, bathy_ds::GeoDataset; varname="elevation", min_water_depth=10.0)

Build an ImmersedBoundaryGrid from a GeoData bathymetry dataset.
"""
function build_immersed_grid_from_geodata(
    grid::LatitudeLongitudeGrid,
    bathy_ds::GeoDataset;
    varname::AbstractString = "elevation",
    min_water_depth::Real = 10.0,
    mask_bay_of_fundy::Bool = false
)
    topo_matrix = GeoData.regrid_2d_field(bathy_ds, varname, 
        collect(Float64, grid.λᶜᵃᵃ)[grid.Hx+1 : grid.Hx+grid.Nx],
        collect(Float64, grid.φᵃᶜᵃ)[grid.Hy+1 : grid.Hy+grid.Ny]
    )
    
    # Apply masking and conditioning
    nx, ny = size(topo_matrix)
    if mask_bay_of_fundy
        target_lons = collect(Float64, grid.λᶜᵃᵃ)[grid.Hx+1 : grid.Hx+grid.Nx]
        target_lats = collect(Float64, grid.φᵃᶜᵃ)[grid.Hy+1 : grid.Hy+grid.Ny]
        for j in 1:ny, i in 1:nx
            if is_in_bay_of_fundy(target_lons[i], target_lats[j])
                topo_matrix[i, j] = 0.0
            end
        end
    end
    
    # Condition bathymetry
    h_floor = Float64(min_water_depth)
    conditioned_topo = smooth_bathymetry(topo_matrix, passes=4, alpha=0.5, h_min=h_floor)
    
    return build_immersed_grid(grid, conditioned_topo)
end

end # module GeoDataIntegration
