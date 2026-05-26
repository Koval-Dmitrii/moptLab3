# import numpy as np
#
# from internal.functions import (
#     QuadraticWellConditioned,
#     QuadraticIllConditioned,
#     Rosenbrock,
#     Ackley,
#     Himmelblau,
# )
# from internal.optimizers import (
#     gradient_descent_momentum,
#     gradient_descent_nesterov,
#     gradient_descent_adagrad,
#     gradient_descent_rmsprop,
#     gradient_descent_adadelta,
#     gradient_descent_adam,
# )
# from internal.utils import (
#     plot_contour_with_trajectory,
#     plot_iter_vs_step,
#     plot_heatmap,
#     print_table,
# )
#
# EPS = 1e-8
#
#
# def _slug(text: str) -> str:
#     return "".join(c.lower() if c.isalnum() else "_" for c in text).strip("_")
#
#
# def _params_to_str(params: dict) -> str:
#     return ", ".join(f"{k}={v}" for k, v in params.items())
#
#
# def _iter_value(result) -> float:
#     return float(result.n_iter) if result.converged else np.nan
#
#
# def _iter_cell(result):
#     return result.n_iter if result.converged else "—"
#
#
# def _status(result) -> str:
#     return "✓" if result.converged else "✗"
#
#
# def _run_method(method_name: str, func, x0: np.ndarray, params: dict, eps: float = EPS):
#     f = func.f
#     g = func.grad
#
#     if method_name == "Momentum":
#         return gradient_descent_momentum(f, g, x0, alpha=params["alpha"], beta=params["beta"], eps=eps)
#     if method_name == "Nesterov":
#         return gradient_descent_nesterov(f, g, x0, alpha=params["alpha"], beta=params["beta"], eps=eps)
#     if method_name == "AdaGrad":
#         return gradient_descent_adagrad(f, g, x0, alpha=params["alpha"], eps=eps)
#     if method_name == "RMSProp":
#         return gradient_descent_rmsprop(f, g, x0, alpha=params["alpha"], rho=params["rho"], eps=eps)
#     if method_name == "AdaDelta":
#         return gradient_descent_adadelta(f, g, x0, rho=params["rho"], eps=eps)
#     if method_name == "Adam":
#         return gradient_descent_adam(
#             f, g, x0, alpha=params["alpha"], beta1=params["beta1"], beta2=params["beta2"], eps=eps
#         )
#
#     raise ValueError(f"Unknown method: {method_name}")
#
#
# def quadratic_parameter_study(save_graphs=True, save_tables=True, eps: float = EPS):
#     print("\n" + "=" * 70)
#     print("Lab3 | Квадратичные функции: зависимость сходимости от параметров")
#     print("=" * 70)
#
#     funcs = [QuadraticWellConditioned(), QuadraticIllConditioned()]
#     x0 = np.array([2.0, 2.0])
#
#     one_param_specs = [
#         ("AdaGrad", "alpha", [0.01, 0.03, 0.05, 0.1, 0.2], "α"),
#         ("AdaDelta", "rho", [0.7, 0.8, 0.9, 0.95, 0.99], "ρ"),
#     ]
#
#     two_param_specs = [
#         ("Momentum", "alpha", [0.001, 0.003, 0.005, 0.01, 0.02], "beta", [0.5, 0.7, 0.85, 0.9, 0.95]),
#         ("Nesterov", "alpha", [0.001, 0.003, 0.005, 0.01, 0.02], "beta", [0.5, 0.7, 0.85, 0.9, 0.95]),
#         ("RMSProp", "alpha", [0.001, 0.003, 0.005, 0.01, 0.02], "rho", [0.7, 0.8, 0.9, 0.95, 0.99]),
#         ("Adam", "beta1", [0.7, 0.8, 0.9, 0.95, 0.99], "beta2", [0.9, 0.95, 0.99, 0.995, 0.999]),
#     ]
#
#     defaults = {
#         "Momentum": {"alpha": 0.005, "beta": 0.9},
#         "Nesterov": {"alpha": 0.005, "beta": 0.9},
#         "AdaGrad": {"alpha": 0.05},
#         "RMSProp": {"alpha": 0.005, "rho": 0.9},
#         "AdaDelta": {"rho": 0.95},
#         "Adam": {"alpha": 0.03, "beta1": 0.9, "beta2": 0.999},
#     }
#
#     for func in funcs:
#         for method_name, param_name, param_values, param_label in one_param_specs:
#             rows = []
#             xs = []
#             ys = []
#
#             for value in param_values:
#                 params = defaults[method_name].copy()
#                 params[param_name] = value
#                 r = _run_method(method_name, func, x0, params, eps=eps)
#                 rows.append([value, _iter_cell(r), f"{r.f_val:.3e}", _status(r)])
#
#                 y = _iter_value(r)
#                 if np.isfinite(y):
#                     xs.append(value)
#                     ys.append(y)
#
#             title = f"{func.name} | {method_name} | {param_name}"
#             print_table([param_label, "Итерации", "f(x*)", "Сошлось?"], rows, title=title, save=save_tables)
#
#             if xs and ys:
#                 plot_iter_vs_step(
#                     xs,
#                     ys,
#                     title=f"{func.name}: {method_name}",
#                     xlabel=f"Параметр {param_label}",
#                     filename=f"fig_quad_line_{_slug(func.name)}_{_slug(method_name)}_{param_name}.png",
#                     save=save_graphs,
#                 )
#
#         for method_name, x_name, x_vals, y_name, y_vals in two_param_specs:
#             rows = []
#             heat = np.full((len(y_vals), len(x_vals)), np.nan)
#
#             for yi, yv in enumerate(y_vals):
#                 for xi, xv in enumerate(x_vals):
#                     params = defaults[method_name].copy()
#                     params[x_name] = xv
#                     params[y_name] = yv
#                     r = _run_method(method_name, func, x0, params, eps=eps)
#                     heat[yi, xi] = _iter_value(r)
#                     rows.append([xv, yv, _iter_cell(r), f"{r.f_val:.3e}", _status(r)])
#
#             title = f"{func.name} | {method_name} | {x_name} x {y_name}"
#             print_table([x_name, y_name, "Итерации", "f(x*)", "Сошлось?"], rows, title=title, save=save_tables)
#
#             plot_heatmap(
#                 heat,
#                 x_vals,
#                 y_vals,
#                 title=f"{func.name}: {method_name}",
#                 xlabel=x_name,
#                 ylabel=y_name,
#                 filename=f"fig_quad_heat_{_slug(func.name)}_{_slug(method_name)}_{x_name}_{y_name}.png",
#                 save=save_graphs,
#             )
#
#
# def quadratic_trajectory_study(save_graphs=True, save_tables=True, eps: float = EPS):
#     print("\n" + "=" * 70)
#     print("Lab3 | Квадратичные функции: траектории с выбранными параметрами")
#     print("=" * 70)
#
#     funcs = [QuadraticWellConditioned(), QuadraticIllConditioned()]
#     x0 = np.array([2.0, 2.0])
#
#     selected = {
#         "Momentum": {"alpha": 0.01, "beta": 0.9},
#         "Nesterov": {"alpha": 0.01, "beta": 0.9},
#         "AdaGrad": {"alpha": 0.2},
#         "RMSProp": {"alpha": 0.01, "rho": 0.9},
#         "AdaDelta": {"rho": 0.95},
#         "Adam": {"alpha": 0.05, "beta1": 0.9, "beta2": 0.999},
#     }
#
#     for func in funcs:
#         rows = []
#         trajectories = {}
#
#         for method_name, params in selected.items():
#             r = _run_method(method_name, func, x0, params, eps=eps)
#             rows.append([method_name, _params_to_str(params), _iter_cell(r), f"{r.f_val:.3e}", _status(r)])
#             trajectories[method_name] = r.trajectory
#
#         print_table(
#             ["Метод", "Параметры", "Итерации", "f(x*)", "Сошлось?"],
#             rows,
#             title=f"Quadratic trajectories | {func.name}",
#             save=save_tables,
#         )
#
#         suffix = "well" if "well" in func.name else "ill"
#         plot_contour_with_trajectory(
#             func.f,
#             title=f"Квадратичная функция: {func.name}",
#             trajectories=trajectories,
#             xlim=(-3, 3),
#             ylim=(-3, 3),
#             filename=f"fig_quad_traj_{suffix}.png",
#             save=save_graphs,
#         )
#
#
# def complex_sensitivity_study(save_graphs=True, save_tables=True, eps: float = EPS):
#     print("\n" + "=" * 70)
#     print("Lab3 | Сложные функции: чувствительность к параметрам")
#     print("=" * 70)
#
#     funcs = [Rosenbrock(), Ackley(), Himmelblau()]
#     x0s = {
#         "Rosenbrock": np.array([-1.0, 1.0]),
#         "Ackley": np.array([1.5, 1.0]),
#         "Himmelblau": np.array([0.0, 0.0]),
#     }
#
#     param_sets = {
#         "Momentum": [
#             {"alpha": 0.001, "beta": 0.8},
#             {"alpha": 0.003, "beta": 0.9},
#             {"alpha": 0.005, "beta": 0.95},
#         ],
#         "Nesterov": [
#             {"alpha": 0.001, "beta": 0.8},
#             {"alpha": 0.003, "beta": 0.9},
#             {"alpha": 0.005, "beta": 0.95},
#         ],
#         "AdaGrad": [
#             {"alpha": 0.01},
#             {"alpha": 0.05},
#             {"alpha": 0.1},
#         ],
#         "RMSProp": [
#             {"alpha": 0.003, "rho": 0.9},
#             {"alpha": 0.005, "rho": 0.95},
#             {"alpha": 0.01, "rho": 0.99},
#         ],
#         "AdaDelta": [
#             {"rho": 0.9},
#             {"rho": 0.95},
#             {"rho": 0.99},
#         ],
#         "Adam": [
#             {"alpha": 0.01, "beta1": 0.85, "beta2": 0.99},
#             {"alpha": 0.02, "beta1": 0.9, "beta2": 0.999},
#             {"alpha": 0.03, "beta1": 0.95, "beta2": 0.999},
#         ],
#     }
#
#     for func in funcs:
#         rows = []
#         x0 = x0s[func.name]
#         for method_name, variants in param_sets.items():
#             for idx, params in enumerate(variants, start=1):
#                 r = _run_method(method_name, func, x0, params, eps=eps)
#                 rows.append([
#                     method_name,
#                     f"set{idx}",
#                     _params_to_str(params),
#                     _iter_cell(r),
#                     f"{r.f_val:.3e}",
#                     _status(r),
#                 ])
#
#         print_table(
#             ["Метод", "Набор", "Параметры", "Итерации", "f(x*)", "Сошлось?"],
#             rows,
#             title=f"Complex sensitivity | {func.name}",
#             save=save_tables,
#         )
#
#
# def complex_trajectory_study(save_graphs=True, save_tables=True, eps: float = EPS):
#     print("\n" + "=" * 70)
#     print("Lab3 | Сложные функции: траектории (2-3 стартовые точки)")
#     print("=" * 70)
#
#     funcs = [Rosenbrock(), Ackley(), Himmelblau()]
#     start_points = {
#         "Rosenbrock": [np.array([-1.0, 1.0]), np.array([0.5, -0.5]), np.array([-1.5, 2.0])],
#         "Ackley": [np.array([1.5, 1.0]), np.array([-2.0, 2.0]), np.array([2.5, -1.5])],
#         "Himmelblau": [np.array([0.0, 0.0]), np.array([-3.0, 3.0]), np.array([3.0, -1.0])],
#     }
#     xlims = {"Rosenbrock": (-2, 2), "Ackley": (-4, 4), "Himmelblau": (-5, 5)}
#     ylims = {"Rosenbrock": (-1, 3), "Ackley": (-4, 4), "Himmelblau": (-5, 5)}
#
#     selected = {
#         "Momentum": {"alpha": 0.003, "beta": 0.9},
#         "Nesterov": {"alpha": 0.003, "beta": 0.9},
#         "AdaGrad": {"alpha": 0.05},
#         "RMSProp": {"alpha": 0.005, "rho": 0.95},
#         "AdaDelta": {"rho": 0.95},
#         "Adam": {"alpha": 0.02, "beta1": 0.9, "beta2": 0.999},
#     }
#
#     for func in funcs:
#         rows = []
#         trajectories = {}
#
#         for x0 in start_points[func.name]:
#             for method_name, params in selected.items():
#                 r = _run_method(method_name, func, x0, params, eps=eps)
#                 key = f"{method_name} x0={np.round(x0, 1)}"
#                 trajectories[key] = r.trajectory
#                 rows.append([
#                     method_name,
#                     str(np.round(x0, 2)),
#                     _params_to_str(params),
#                     _iter_cell(r),
#                     f"{r.f_val:.3e}",
#                     _status(r),
#                 ])
#
#         print_table(
#             ["Метод", "x0", "Параметры", "Итерации", "f(x*)", "Сошлось?"],
#             rows,
#             title=f"Complex trajectories | {func.name}",
#             save=save_tables,
#         )
#
#         plot_contour_with_trajectory(
#             func.f,
#             title=f"Сложная функция: {func.name}",
#             trajectories=trajectories,
#             xlim=xlims[func.name],
#             ylim=ylims[func.name],
#             filename=f"fig_complex_traj_{_slug(func.name)}.png",
#             save=save_graphs,
#         )


import numpy as np

from internal.functions import (
    QuadraticWellConditioned,
    QuadraticIllConditioned,
    Rosenbrock,
    Ackley,
    Himmelblau,
)
from internal.optimizers import (
    gradient_descent_momentum,
    gradient_descent_nesterov,
    gradient_descent_adagrad,
    gradient_descent_rmsprop,
    gradient_descent_adadelta,
    gradient_descent_adam,
)
from internal.utils import (
    plot_contour_with_trajectory,
    plot_iter_vs_step,
    plot_heatmap,
    print_table,
)

EPS = 1e-8


def _slug(text: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in text).strip("_")


def _params_to_str(params: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in params.items())


def _iter_value(result) -> float:
    return float(result.n_iter) if result.converged else np.nan


def _iter_cell(result):
    return result.n_iter if result.converged else "—"


def _status(result) -> str:
    return "✓" if result.converged else "✗"


def _run_method(method_name: str, func, x0: np.ndarray, params: dict, eps: float = EPS):
    f = func.f
    g = func.grad

    if method_name == "Momentum":
        return gradient_descent_momentum(f, g, x0, alpha=params["alpha"], beta=params["beta"], eps=eps)
    if method_name == "Nesterov":
        return gradient_descent_nesterov(f, g, x0, alpha=params["alpha"], beta=params["beta"], eps=eps)
    if method_name == "AdaGrad":
        return gradient_descent_adagrad(f, g, x0, alpha=params["alpha"], eps=eps)
    if method_name == "RMSProp":
        return gradient_descent_rmsprop(f, g, x0, alpha=params["alpha"], rho=params["rho"], eps=eps)
    if method_name == "AdaDelta":
        return gradient_descent_adadelta(f, g, x0, rho=params["rho"], eps=eps)
    if method_name == "Adam":
        return gradient_descent_adam(
            f, g, x0, alpha=params["alpha"], beta1=params["beta1"], beta2=params["beta2"], eps=eps
        )

    raise ValueError(f"Unknown method: {method_name}")


# ─────────────────────────────────────────────────────────────────────────────
# Задание 1: Исследование параметров на квадратичных функциях
# ─────────────────────────────────────────────────────────────────────────────

def quadratic_parameter_study(save_graphs=True, save_tables=True, eps: float = EPS):
    print("\n" + "=" * 70)
    print("Lab3 | Квадратичные функции: зависимость сходимости от параметров")
    print("=" * 70)

    funcs = [QuadraticWellConditioned(), QuadraticIllConditioned()]
    x0 = np.array([2.0, 2.0])

    # Методы с одним исследуемым параметром → line-графики
    one_param_specs = [
        ("AdaGrad",  "alpha", [0.01, 0.05, 0.1, 0.2, 0.5],        "α"),
        ("AdaDelta", "rho",   [0.7,  0.8,  0.9, 0.95, 0.99],       "ρ"),
    ]

    # Методы с двумя параметрами → heatmap
    two_param_specs = [
        ("Momentum", "alpha", [0.001, 0.003, 0.005, 0.01, 0.02],
                     "beta",  [0.5,   0.7,   0.85,  0.9,  0.95]),
        ("Nesterov", "alpha", [0.001, 0.003, 0.005, 0.01, 0.02],
                     "beta",  [0.5,   0.7,   0.85,  0.9,  0.95]),
        ("RMSProp",  "alpha", [0.001, 0.003, 0.005, 0.01, 0.02],
                     "rho",   [0.7,   0.8,   0.9,   0.95, 0.99]),
        ("Adam",     "beta1", [0.7,   0.8,   0.9,   0.95, 0.99],
                     "beta2", [0.9,   0.95,  0.99,  0.995, 0.999]),
    ]

    # Значения по умолчанию при фиксации одного параметра
    defaults = {
        "Momentum": {"alpha": 0.005, "beta": 0.9},
        "Nesterov": {"alpha": 0.005, "beta": 0.9},
        "AdaGrad":  {"alpha": 0.1},
        "RMSProp":  {"alpha": 0.01,  "rho":  0.9},
        "AdaDelta": {"rho":   0.95},
        "Adam":     {"alpha": 0.05,  "beta1": 0.9, "beta2": 0.999},
    }

    for func in funcs:
        # --- line-графики ---
        for method_name, param_name, param_values, param_label in one_param_specs:
            rows, xs, ys = [], [], []
            for value in param_values:
                params = {**defaults[method_name], param_name: value}
                r = _run_method(method_name, func, x0, params, eps=eps)
                rows.append([value, _iter_cell(r), f"{r.f_val:.3e}", _status(r)])
                if r.converged:
                    xs.append(value)
                    ys.append(float(r.n_iter))

            title = f"{func.name} | {method_name} | {param_name}"
            print_table([param_label, "Итерации", "f(x*)", "Сошлось?"], rows,
                        title=title, save=save_tables)
            if len(xs) >= 2:
                plot_iter_vs_step(
                    xs, ys,
                    title=f"{func.name}: {method_name}",
                    xlabel=f"Параметр {param_label}",
                    filename=f"fig_quad_line_{_slug(func.name)}_{_slug(method_name)}_{param_name}.png",
                    save=save_graphs,
                )

        # --- heatmap ---
        for method_name, x_name, x_vals, y_name, y_vals in two_param_specs:
            rows = []
            heat = np.full((len(y_vals), len(x_vals)), np.nan)
            for yi, yv in enumerate(y_vals):
                for xi, xv in enumerate(x_vals):
                    params = {**defaults[method_name], x_name: xv, y_name: yv}
                    r = _run_method(method_name, func, x0, params, eps=eps)
                    heat[yi, xi] = _iter_value(r)
                    rows.append([xv, yv, _iter_cell(r), f"{r.f_val:.3e}", _status(r)])

            title = f"{func.name} | {method_name} | {x_name} x {y_name}"
            print_table([x_name, y_name, "Итерации", "f(x*)", "Сошлось?"], rows,
                        title=title, save=save_tables)
            plot_heatmap(
                heat, x_vals, y_vals,
                title=f"{func.name}: {method_name}",
                xlabel=x_name, ylabel=y_name,
                filename=f"fig_quad_heat_{_slug(func.name)}_{_slug(method_name)}_{x_name}_{y_name}.png",
                save=save_graphs,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Задание 2: Траектории на квадратичных функциях
# ─────────────────────────────────────────────────────────────────────────────

def quadratic_trajectory_study(save_graphs=True, save_tables=True, eps: float = EPS):
    print("\n" + "=" * 70)
    print("Lab3 | Квадратичные функции: траектории с выбранными параметрами")
    print("=" * 70)

    funcs = [QuadraticWellConditioned(), QuadraticIllConditioned()]
    x0 = np.array([2.0, 2.0])

    # Один рабочий набор для каждого метода (проверено на обеих квадратиках)
    selected = {
        "Momentum": {"alpha": 0.01,  "beta":  0.9},
        "Nesterov": {"alpha": 0.01,  "beta":  0.9},
        "AdaGrad":  {"alpha": 0.2},
        "RMSProp":  {"alpha": 0.01,  "rho":   0.9},
        "AdaDelta": {"rho":   0.95},
        "Adam":     {"alpha": 0.05,  "beta1": 0.9, "beta2": 0.999},
    }

    for func in funcs:
        rows, trajectories = [], {}
        for method_name, params in selected.items():
            r = _run_method(method_name, func, x0, params, eps=eps)
            rows.append([method_name, _params_to_str(params),
                         _iter_cell(r), f"{r.f_val:.3e}", _status(r)])
            trajectories[method_name] = r.trajectory

        print_table(["Метод", "Параметры", "Итерации", "f(x*)", "Сошлось?"], rows,
                    title=f"Quadratic trajectories | {func.name}", save=save_tables)

        suffix = "well" if "well" in func.name else "ill"
        plot_contour_with_trajectory(
            func.f,
            title=f"Квадратичная функция: {func.name}",
            trajectories=trajectories,
            xlim=(-3, 3), ylim=(-3, 3),
            filename=f"fig_quad_traj_{suffix}.png",
            save=save_graphs,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Задание 3: Чувствительность к параметрам на сложных функциях
# ─────────────────────────────────────────────────────────────────────────────

def complex_sensitivity_study(save_graphs=True, save_tables=True, eps: float = EPS):
    print("\n" + "=" * 70)
    print("Lab3 | Сложные функции: чувствительность к параметрам")
    print("=" * 70)

    funcs = [Rosenbrock(), Ackley(), Himmelblau()]
    x0s = {
        "Rosenbrock": np.array([-1.0,  1.0]),
        "Ackley":     np.array([ 1.5,  1.0]),
        "Himmelblau": np.array([ 0.0,  0.0]),
    }

    # 3 набора параметров для каждого метода.
    # Подобраны так, чтобы хотя бы один набор сходился на каждой функции;
    # несходимость RMSProp/AdaDelta на Розенброке — честный результат.
    param_sets = {
        "Momentum": [
            {"alpha": 0.001, "beta": 0.8},
            {"alpha": 0.003, "beta": 0.9},
            {"alpha": 0.001, "beta": 0.9},
        ],
        "Nesterov": [
            {"alpha": 0.001,  "beta": 0.8},
            {"alpha": 0.001,  "beta": 0.9},
            {"alpha": 0.0005, "beta": 0.9},
        ],
        "AdaGrad": [
            {"alpha": 0.05},
            {"alpha": 0.1},
            {"alpha": 0.2},
        ],
        # RMSProp сходится на Химмельблау; на Розенброке и Экли — не сходится
        # (иллюстрирует ограничения метода на невыпуклых задачах)
        "RMSProp": [
            {"alpha": 0.02,  "rho": 0.9},
            {"alpha": 0.05,  "rho": 0.9},
            {"alpha": 0.01,  "rho": 0.95},
        ],
        # AdaDelta аналогично: на Экли и Химмельблау при rho=0.99 сходится
        "AdaDelta": [
            {"rho": 0.99},
            {"rho": 0.999},
            {"rho": 0.95},
        ],
        "Adam": [
            {"alpha": 0.01,  "beta1": 0.85, "beta2": 0.99},
            {"alpha": 0.02,  "beta1": 0.9,  "beta2": 0.999},
            {"alpha": 0.05,  "beta1": 0.9,  "beta2": 0.999},
        ],
    }

    for func in funcs:
        rows = []
        x0 = x0s[func.name]
        for method_name, variants in param_sets.items():
            for idx, params in enumerate(variants, start=1):
                r = _run_method(method_name, func, x0, params, eps=eps)
                rows.append([
                    method_name, f"set{idx}",
                    _params_to_str(params),
                    _iter_cell(r), f"{r.f_val:.3e}", _status(r),
                ])

        print_table(
            ["Метод", "Набор", "Параметры", "Итерации", "f(x*)", "Сошлось?"],
            rows,
            title=f"Complex sensitivity | {func.name}",
            save=save_tables,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Задание 4: Траектории на сложных функциях (2–3 стартовые точки)
# ─────────────────────────────────────────────────────────────────────────────

def complex_trajectory_study(save_graphs=True, save_tables=True, eps: float = EPS):
    print("\n" + "=" * 70)
    print("Lab3 | Сложные функции: траектории (2–3 стартовые точки)")
    print("=" * 70)

    funcs = [Rosenbrock(), Ackley(), Himmelblau()]

    start_points = {
        "Rosenbrock": [np.array([-1.0,  1.0]),
                       np.array([ 0.5, -0.5]),
                       np.array([-1.5,  2.0])],
        "Ackley":     [np.array([ 1.5,  1.0]),
                       np.array([-2.0,  2.0]),
                       np.array([ 2.5, -1.5])],
        "Himmelblau": [np.array([ 0.0,  0.0]),
                       np.array([-3.0,  3.0]),
                       np.array([ 3.0, -1.0])],
    }
    xlims = {"Rosenbrock": (-2, 2),  "Ackley": (-4, 4), "Himmelblau": (-5, 5)}
    ylims = {"Rosenbrock": (-1, 3),  "Ackley": (-4, 4), "Himmelblau": (-5, 5)}

    # Один набор на метод; выбран так, чтобы давать наибольшее число
    # сходящихся запусков суммарно по всем функциям и стартовым точкам.
    # RMSProp: alpha=0.02, rho=0.9 сходится на Химмельблау со всех x0,
    #          на Розенброке не сходится — это честная иллюстрация ограничений.
    # AdaDelta: rho=0.99 сходится на Экли и Химмельблау.
    selected = {
        "Momentum": {"alpha": 0.003, "beta":  0.9},
        "Nesterov": {"alpha": 0.001, "beta":  0.9},
        "AdaGrad":  {"alpha": 0.1},
        "RMSProp":  {"alpha": 0.02,  "rho":   0.9},
        "AdaDelta": {"rho":   0.99},
        "Adam":     {"alpha": 0.02,  "beta1": 0.9, "beta2": 0.999},
    }

    for func in funcs:
        rows, trajectories = [], {}
        for x0 in start_points[func.name]:
            for method_name, params in selected.items():
                r = _run_method(method_name, func, x0, params, eps=eps)
                key = f"{method_name} x0={np.round(x0, 1)}"
                trajectories[key] = r.trajectory
                rows.append([
                    method_name,
                    str(np.round(x0, 2)),
                    _params_to_str(params),
                    _iter_cell(r), f"{r.f_val:.3e}", _status(r),
                ])

        print_table(
            ["Метод", "x0", "Параметры", "Итерации", "f(x*)", "Сошлось?"],
            rows,
            title=f"Complex trajectories | {func.name}",
            save=save_tables,
        )

        plot_contour_with_trajectory(
            func.f,
            title=f"Сложная функция: {func.name}",
            trajectories=trajectories,
            xlim=xlims[func.name],
            ylim=ylims[func.name],
            filename=f"fig_complex_traj_{_slug(func.name)}.png",
            save=save_graphs,
        )