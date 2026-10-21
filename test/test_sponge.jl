# ==============================================================================
# GeoOceanModel Test: Lateral Sponge Relaxation
# ==============================================================================

using Test
using Oceananigans
using GeoOceanModel

@testset "Sponge Relaxation" begin
    @testset "CartesianSpongeMask bitstype & weights" begin
        mask = CartesianSpongeMask(
            -68.0, -56.0, 40.0, 48.0, 0.5,
            true, false, false, true
        )
        @test isbitstype(typeof(mask))

        # Interior point: zero weight
        @test mask(-62.0, 44.0, 0.0) == 0.0

        # Active southern boundary: full damping
        @test mask(-62.0, 40.0, 0.0) == 1.0

        # Active eastern boundary: full damping
        @test mask(-56.0, 44.0, 0.0) == 1.0

        # Inactive northern and western boundaries: zero weight
        @test mask(-62.0, 48.0, 0.0) == 0.0
        @test mask(-68.0, 44.0, 0.0) == 0.0

        # Quadratic taper: half-width along southern boundary
        # ((lat_min + sponge_width - y) / sponge_width)^2 = ((40.0 + 0.5 - 40.25) / 0.5)^2 = 0.25
        @test isapprox(mask(-62.0, 40.25, 0.0), 0.25, atol = 1e-6)
    end

    @testset "CartesianSpongeMask inflow variant ignores inactive edges" begin
        # The inflow mask is the same functor with only the east edge active: the
        # southern radiation edge must carry no inflow weight.
        inflow = CartesianSpongeMask(
            -68.0, -56.0, 40.0, 48.0, 0.5,
            true, false, false, false
        )
        @test isbitstype(typeof(inflow))

        # Southern edge: no inflow weight
        @test inflow(-62.0, 40.0, 0.0) == 0.0

        # Eastern edge: full inflow weight
        @test inflow(-56.0, 44.0, 0.0) == 1.0

        # Interior: zero
        @test inflow(-62.0, 44.0, 0.0) == 0.0
    end
end
