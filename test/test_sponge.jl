# ==============================================================================
# GeoOceanModel Test: Lateral Sponge Relaxation & Rotated Grid Boundary Dampening
# ==============================================================================

using Test
using Oceananigans
using Oceananigans.OrthogonalSphericalShellGrids: RotatedLatitudeLongitudeGrid
using GeoOceanModel

@testset "Sponge Relaxation & Rotated Boundary Layer" begin
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

    @testset "RotatedSpongeMask inverse rotation & boundary alignment" begin
        # 50° rotated pole for Scotian Shelf
        np = (167.38, 33.44)
        rot_lon = (-5.0, 5.0)
        rot_lat = (-1.0, 1.0)
        width = 0.45

        mask_rot = RotatedSpongeMask(
            np[1], np[2], rot_lon, rot_lat, width;
            active_east = true, active_west = false,
            active_north = false, active_south = true
        )
        @test isbitstype(typeof(mask_rot))

        # Center point (0, 0 in rotated): zero weight
        # Find geographic center using rotate_to_curvilinear consistency
        dλ = deg2rad(np[1])
        dφ = π / 2.0 - deg2rad(np[2])
        cos_dλ = cos(dλ)
        sin_dλ = sin(dλ)
        cos_dφ = cos(dφ)
        sin_dφ = sin(dφ)

        # Check a point outside rotated domain on south
        # Point on south boundary (φ' = -1.0) should have gamma_s = 1.0
        # Create a small grid to query true geographic coordinates of the boundary
        g = RotatedLatitudeLongitudeGrid(
            CPU();
            size = (10, 10, 2),
            longitude = rot_lon,
            latitude = rot_lat,
            z = (-100.0, 0.0),
            north_pole = np
        )

        bounds = extract_rotated_grid_bounds(g, np)
        @test isapprox(bounds[1][1], rot_lon[1], atol = 1e-4)
        @test isapprox(bounds[1][2], rot_lon[2], atol = 1e-4)
        @test isapprox(bounds[2][1], rot_lat[1], atol = 1e-4)
        @test isapprox(bounds[2][2], rot_lat[2], atol = 1e-4)

        # Test south edge: for all i along j=1, mask must evaluate to 1.0
        for i in 1:10
            x_geo = g.λᶜᶜᵃ[i, 1]
            y_geo = g.φᶜᶜᵃ[i, 1]
            # Since j=1 center is slightly inside boundary:
            # φ'_center = -1.0 + 0.5 * 0.2 = -0.9, which is within width 0.45
            # Thus gamma must be strictly positive
            w = mask_rot(x_geo, y_geo, 0.0)
            @test w > 0.0
        end

        # Interior center point (j=5) must have zero weight
        for i in 3:8
            x_geo = g.λᶜᶜᵃ[i, 5]
            y_geo = g.φᶜᶜᵃ[i, 5]
            @test mask_rot(x_geo, y_geo, 0.0) == 0.0
        end

        # Inflow sponge mask (active_east only)
        mask_inflow = RotatedSpongeMask(
            np[1], np[2], rot_lon, rot_lat, width;
            active_east = true, active_west = false,
            active_north = false, active_south = false
        )
        @test isbitstype(typeof(mask_inflow))

        # Southern boundary must have zero inflow weight
        for i in 1:10
            x_geo = g.λᶜᶜᵃ[i, 1]
            y_geo = g.φᶜᶜᵃ[i, 1]
            @test mask_inflow(x_geo, y_geo, 0.0) == 0.0
        end

        # Eastern boundary face (i=11) must have positive inflow weight
        for j in 1:10
            x_geo = g.λᶠᶜᵃ[11, j]
            y_geo = g.φᶠᶜᵃ[11, j]
            @test mask_inflow(x_geo, y_geo, 0.0) > 0.0
        end
    end
end
