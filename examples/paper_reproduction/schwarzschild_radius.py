"""Example: Reproducing Schwarzschild radius calculation.

This example demonstrates how to use dimtensor's research module to
reproduce the classic Schwarzschild radius calculation for black holes.

References:
    K. Schwarzschild (1916). "Über das Gravitationsfeld eines Massenpunktes
    nach der Einsteinschen Theorie". Sitzungsberichte der Königlich
    Preussischen Akademie der Wissenschaften. 1: 189–196.
"""

from dimtensor import DimArray, units
from dimtensor.constants import G, c
from dimtensor.research import (
    Paper,
    ReproductionResult,
    compare_values,
    generate_report,
)


def main():
    """Reproduce Schwarzschild radius calculation."""

    # 1. Create paper metadata
    print("Creating paper metadata...")
    paper = Paper(
        title="Schwarzschild Radius for Stellar Objects",
        authors=["K. Schwarzschild"],
        doi="10.1002/asna.19160200402",
        year=1916,
        journal="Astronomische Nachrichten",
        abstract=(
            "Calculation of the Schwarzschild radius for various stellar objects "
            "using the formula r_s = 2GM/c²."
        ),
        assumptions=[
            "Spherical symmetry",
            "Static metric",
            "Non-rotating black hole",
        ],
        tags=["general relativity", "black holes", "astrophysics"],
    )

    # Add published values (modern accepted values)
    paper.add_published_value(
        "solar_schwarzschild_radius",
        DimArray(2.95, units.km),
        "km",
    )
    paper.add_published_value(
        "earth_schwarzschild_radius",
        DimArray(8.87e-3, units.m),
        "mm",
    )

    print(f"Paper: {paper}")
    print()

    # 2. Compute values using dimtensor
    print("Computing Schwarzschild radii...")

    # Solar mass: 1.989 × 10³⁰ kg
    solar_mass = DimArray(1.989e30, units.kg)
    r_s_solar = 2 * G * solar_mass / c**2

    # Earth mass: 5.972 × 10²⁴ kg
    earth_mass = DimArray(5.972e24, units.kg)
    r_s_earth = 2 * G * earth_mass / c**2

    print(f"Computed solar Schwarzschild radius: {r_s_solar.to(units.km)}")
    print(f"Computed Earth Schwarzschild radius: {r_s_earth.to(units.m)}")
    print()

    # 3. Create reproduction result
    result = ReproductionResult(
        paper=paper,
        reproducer="dimtensor example",
        code_repository="https://github.com/marcoloco23/dimtensor",
    )

    result.add_computed_value("solar_schwarzschild_radius", r_s_solar.to(units.km))
    result.add_computed_value("earth_schwarzschild_radius", r_s_earth.to(units.m))

    # 4. Compare values
    print("Comparing published vs computed values...")

    comparison_solar = compare_values(
        paper.published_values["solar_schwarzschild_radius"],
        r_s_solar.to(units.km),
        rtol=0.01,  # 1% tolerance
        quantity_name="solar_schwarzschild_radius",
    )

    comparison_earth = compare_values(
        paper.published_values["earth_schwarzschild_radius"],
        r_s_earth.to(units.m),
        rtol=0.01,  # 1% tolerance
        quantity_name="earth_schwarzschild_radius",
    )

    result.add_comparison("solar_schwarzschild_radius", comparison_solar)
    result.add_comparison("earth_schwarzschild_radius", comparison_earth)

    print(f"Solar radius match: {comparison_solar.matches}")
    print(f"  Relative error: {comparison_solar.relative_error:.2e}")
    print(f"Earth radius match: {comparison_earth.matches}")
    print(f"  Relative error: {comparison_earth.relative_error:.2e}")
    print()

    # 5. Print summary
    print("Reproduction summary:")
    summary = result.summary()
    print(f"  Status: {summary['status'].upper()}")
    print(f"  Quantities compared: {summary['num_compared']}")
    print(f"  Matches: {summary['num_matches']}")
    if summary.get('mean_relative_error'):
        print(f"  Mean relative error: {summary['mean_relative_error']:.2e}")
    print()

    # 6. Generate report
    print("Generating markdown report...")
    report = generate_report(result, format="markdown")

    # Save to file
    with open("schwarzschild_report.md", "w") as f:
        f.write(report)

    print("Report saved to: schwarzschild_report.md")
    print()

    # 7. Save result to JSON
    result.to_json("schwarzschild_result.json")
    print("Result saved to: schwarzschild_result.json")

    # Print first few lines of report
    print("\n" + "=" * 70)
    print("Report preview:")
    print("=" * 70)
    print("\n".join(report.split("\n")[:25]))
    print("...")


if __name__ == "__main__":
    main()
