from internal.tasks import (
    quadratic_parameter_study,
    quadratic_trajectory_study,
    complex_sensitivity_study,
    complex_trajectory_study,
)
from internal.runner import run


def main():
    run([
        quadratic_parameter_study,
        quadratic_trajectory_study,
        complex_sensitivity_study,
        complex_trajectory_study,
    ], output_root="results", save_graphs=True, save_tables=True)


if __name__ == "__main__":
    main()
