
using Oceananigans
using Oceananigans.Architectures: AbstractArchitecture, CPU, GPU
using CUDA
using Preferences

"""
    resolve_architecture(arch::Union{Symbol, String, Bool, AbstractArchitecture} = :cpu; 
                         fallback_to_cpu::Bool = false) -> AbstractArchitecture

Resolve and instantiate the computational architecture (`CPU()` or `GPU(...)`) for 
Oceananigans hydrodynamic simulations and particle tracking routines.
"""
function resolve_architecture(
    arch::Union{Symbol, String, Bool, AbstractArchitecture} = :cpu; 
    fallback_to_cpu::Bool = false
)::AbstractArchitecture
    if arch isa AbstractArchitecture 
        return arch 
    end

    is_gpu_req = (arch === true) || (arch === :gpu) || (arch === :cuda) || 
                 (arch isa String && lowercase(arch) in ["gpu", "cuda"])
    
    if !is_gpu_req 
        return CPU() 
    end

    # Check CUDA availability using modern CUDA.jl API
    cuda_functional = CUDA.functional()
    
    if !cuda_functional
        if fallback_to_cpu
            @warn "CUDA GPU hardware was requested, but no functional NVIDIA CUDA environment " *
                  "was detected (ensure the CUDA.jl package is installed and NVIDIA drivers " *
                  "are accessible). Falling back to CPU()."
            return CPU()
        else
            error(
                "CUDA GPU acceleration was requested (`arch = $(arch)`), but no functional " *
                "NVIDIA CUDA driver or GPU device was detected on this system.\n" *
                "To execute on CPU, set `architecture = :cpu` (or `--cpu` CLI flag).\n" *
                "To enable automatic fallback, set `fallback_to_cpu = true`."
            )
        end
    end

    # Create GPU backend - use Oceananigans' GPU constructor which handles backend creation
    try
        return GPU()
    catch err
        if fallback_to_cpu
            @warn "Failed to initialize GPU backend: $(err). Falling back to CPU()."
            return CPU()
        else
            error("Failed to initialize GPU backend: $(err)")
        end
    end
end