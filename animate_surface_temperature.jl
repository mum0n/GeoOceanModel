using Pkg
Pkg.activate("C:/Users/Nathalie/projects/GeoOceanModel")

using GeoOceanModel

# =============================================================================
# CONFIGURATION — edit these values as needed
# =============================================================================
const STORE_PATH = "C:/Users/Nathalie/projects/lakehouse/GeoOceanModel/scotian_shelf/hydrodynamics.zarr"
const OUTPUT_DIR = "C:/Users/Nathalie/projects/lakehouse/GeoOceanModel/scotian_shelf/outputs"
const OUTPUT_PATH = joinpath(OUTPUT_DIR, "temperature_animation.mp4")

const VARIABLE = :temperature    # :temperature, :salinity, :u, :v, :w, :speed
const DEPTH_LEVEL = 1            # 1 = surface, 2 = next level down, etc.
const FPS = 2                    # frames per second
const N_FRAMES = nothing         # nothing = use all available time steps

# =============================================================================

mkpath(OUTPUT_DIR)

println("="^60)
println("GeoOceanModel — Hydrodynamic Animation Generator")
println("="^60)
println("Store:      $STORE_PATH")
println("Variable:   $VARIABLE")
println("Depth:      level $DEPTH_LEVEL")
println("FPS:        $FPS")
println("Output:     $OUTPUT_PATH")
println("="^60)

# --- Use the library's animation function ---
println("\nLoading hydrodynamic store and recording animation...")
t_start = time()

anim_path = GeoOceanModel.animate_hydrodynamic_field(
    STORE_PATH;
    variable = VARIABLE,
    depth_level = DEPTH_LEVEL,
    n_frames = N_FRAMES,
    framerate = FPS,
    output_path = OUTPUT_PATH,
    title = "Scotian Shelf — Surface Temperature"
)

t_elapsed = time() - t_start
println("\n" * "="^60)
println("Animation complete!")
println("Output:   $anim_path")
println("Variable: $VARIABLE")
println("Depth:    level $DEPTH_LEVEL (surface)")
println("Time:     $(round(t_elapsed, digits=1)) s")
println("="^60)
