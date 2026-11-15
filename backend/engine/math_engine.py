import re
from typing import Any, Dict, List, Optional
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


def _sample_function_points(expr: sp.Expr, var: sp.Symbol, x_min: float = -8.0, x_max: float = 8.0, steps: int = 121) -> Optional[Dict[str, Any]]:
    try:
        deriv = sp.diff(expr, var)
        f_num = sp.lambdify(var, expr, modules=["numpy"])
        df_num = sp.lambdify(var, deriv, modules=["numpy"])
        xs = np.linspace(x_min, x_max, steps)
        points: List[Dict[str, Optional[float]]] = []
        valid_ys: List[float] = []

        for xv in xs:
            try:
                yv = float(np.real(f_num(float(xv))))
                dyv = float(np.real(df_num(float(xv))))
                if not np.isfinite(yv) or abs(yv) > 1e5:
                    yv_clean = None
                else:
                    yv_clean = round(yv, 4)
                    valid_ys.append(yv_clean)
                dy_clean = round(dyv, 4) if (np.isfinite(dyv) and abs(dyv) <= 1e5) else None
                points.append({"x": round(float(xv), 3), "y": yv_clean, "dy": dy_clean})
            except Exception:
                points.append({"x": round(float(xv), 3), "y": None, "dy": None})

        if len(valid_ys) < 10:
            return None

        return {
            "type": "function_plot",
            "title": f"f(x) = {sp.sstr(expr)}",
            "expression": str(expr),
            "derivative": str(deriv),
            "points": points,
            "y_min": round(min(valid_ys), 3),
            "y_max": round(max(valid_ys), 3),
        }
    except Exception:
        return None


def _extract_math_expression(text: str) -> str:
    cleaned = text.strip().rstrip("?.!")
    # Remove natural language prefixes
    cleaned = re.sub(
        r"^(?:please\s+|can\s+you\s+|could\s+you\s+|xyvora\s+)?(?:differentiate\s+and\s+plot|plot\s+and\s+differentiate|find\s+the\s+derivative\s+of|take\s+the\s+derivative\s+of|differentiate|derive|find\s+the\s+integral\s+of|integrate|solve\s+for\s+x\s+in|solve|factor|simplify|expand|plot\s+the\s+function|graph\s+the\s+function|plot|graph|evaluate|calculate|compute|what\s+is)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    # Remove trailing clauses like "and plot it", "and find its derivative", "for x"
    cleaned = re.sub(
        r"\s+(?:and\s+plot(?:\s+it)?|and\s+graph(?:\s+it)?|and\s+find\s+its\s+derivative|and\s+differentiate(?:\s+it)?|with\s+respect\s+to\s+x|for\s+x)$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"^f\(x\)\s*=\s*|^y\s*=\s*", "", cleaned.strip(), flags=re.IGNORECASE)
    return cleaned.strip()


class SymbolicMathEngine:
    def solve_query(self, query: str) -> Dict[str, Any]:
        q_strip = query.strip()
        q_low = q_strip.lower()
        x = sp.Symbol("x", real=True)
        local_dict = {
            "x": x,
            "pi": sp.pi,
            "e": sp.E,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "exp": sp.exp,
            "ln": sp.log,
            "log": sp.log,
            "sqrt": sp.sqrt,
            "abs": sp.Abs,
        }

        # 1. Percentage pattern: "18.75% of 14500"
        pct_match = re.search(r"([\-+]?\d+(?:\.\d+)?)\s*%\s*of\s*([\-+]?\d+(?:,\d{3})*(?:\.\d+)?)", q_low)
        if pct_match:
            pct = float(pct_match.group(1))
            base = float(pct_match.group(2).replace(",", ""))
            ans = (pct / 100.0) * base
            return {
                "ok": True,
                "operation": "percentage",
                "input": f"{pct}% of {base:g}",
                "result": f"{ans:,.6g}",
                "steps": [
                    f"Convert percentage to decimal: `{pct}% = {pct / 100.0:g}`",
                    f"Multiply by base value: `{pct / 100.0:g} × {base:g} = {ans:,.6g}`",
                ],
                "artifacts": [],
            }

        # 2. Descriptive statistics: "mean median std of 12, 18, 25..."
        if any(k in q_low for k in ("mean", "median", "standard deviation", "std", "variance", "average")):
            nums = [float(n) for n in re.findall(r"[\-+]?\d+(?:\.\d+)?", q_strip)]
            if len(nums) >= 2:
                arr = np.array(nums, dtype=float)
                mean_v = float(np.mean(arr))
                med_v = float(np.median(arr))
                std_v = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
                var_v = float(np.var(arr, ddof=1)) if len(arr) > 1 else 0.0
                chart_artifact = {
                    "type": "chart",
                    "chart_type": "bar",
                    "title": "Dataset Distribution & Summary",
                    "data": [{"label": f"#{i+1}", "value": round(v, 3)} for i, v in enumerate(nums[:25])],
                }
                return {
                    "ok": True,
                    "operation": "statistics",
                    "input": f"Dataset (n = {len(nums)}): {nums}",
                    "result": f"Mean = {mean_v:.4g}, Median = {med_v:.4g}, Sample Std Dev = {std_v:.4g}",
                    "steps": [
                        f"**Count ($n$)**: `{len(nums)}` | **Sum**: `{np.sum(arr):.4g}` | **Range**: `[{np.min(arr):.4g}, {np.max(arr):.4g}]`",
                        f"**Arithmetic Mean**: `{mean_v:.6g}`",
                        f"**Median**: `{med_v:.6g}`",
                        f"**Sample Variance ($s^2$)**: `{var_v:.6g}` | **Sample Std Dev ($s$)**: `{std_v:.6g}`",
                    ],
                    "artifacts": [chart_artifact],
                }

        # 3. Matrix / Eigenvalues: "[[4, 2], [1, 3]]"
        mat_match = re.search(r"(\[\s*\[.*?\]\s*\])", q_strip)
        if mat_match:
            try:
                import ast
                raw_mat = ast.literal_eval(mat_match.group(1))
                M = sp.Matrix(raw_mat)
                det = M.det()
                eigenvals = M.eigenvals()
                eig_str = ", ".join(f"λ = {sp.sstr(k)} (multiplicity {v})" for k, v in eigenvals.items())
                steps = [
                    f"**Matrix Dimensions**: `{M.rows} × {M.cols}`",
                    f"**Trace**: `{M.trace()}` | **Determinant**: `{det}`",
                    f"**Eigenvalues**: `{eig_str}`",
                ]
                if det != 0 and M.rows == M.cols:
                    steps.append(f"**Inverse $M^{{-1}}$**: `{M.inv().tolist()}`")
                return {
                    "ok": True,
                    "operation": "linear_algebra",
                    "input": str(raw_mat),
                    "result": f"det = {det} | Eigenvalues: {eig_str}",
                    "steps": steps,
                    "artifacts": [],
                }
            except Exception as exc:
                return {"ok": False, "error": str(exc), "artifacts": []}

        # 4. Definite integral pattern: "integrate x^2 from 0 to 4"
        def_int_match = re.search(r"integrate\s+(.+?)\s+from\s+([\-+0-9.pi/]+)\s+to\s+([\-+0-9.pi/]+)", q_low)
        if def_int_match:
            try:
                expr_s, a_s, b_s = def_int_match.group(1), def_int_match.group(2), def_int_match.group(3)
                expr = parse_expr(expr_s, local_dict=local_dict, transformations=TRANSFORMATIONS)
                a = parse_expr(a_s, local_dict=local_dict, transformations=TRANSFORMATIONS)
                b = parse_expr(b_s, local_dict=local_dict, transformations=TRANSFORMATIONS)
                indef = sp.integrate(expr, x)
                val = sp.integrate(expr, (x, a, b))
                plot = _sample_function_points(expr, x, float(sp.N(a)) - 1.0, float(sp.N(b)) + 1.0)
                return {
                    "ok": True,
                    "operation": "definite_integral",
                    "input": f"∫ from {a} to {b} of ({expr}) dx",
                    "result": f"{val} (≈ {float(sp.N(val)):.6g})",
                    "steps": [
                        f"**Antiderivative $F(x)$**: `{indef} + C`",
                        f"**Evaluate $F({b}) - F({a})$**: `{sp.subs(indef, x, b) if hasattr(sp, 'subs') else indef.subs(x, b)} - ({indef.subs(x, a)})`",
                        f"**Exact Definite Integral**: `{val}` (`≈ {float(sp.N(val)):.6g}`)",
                    ],
                    "artifacts": [plot] if plot else [],
                }
            except Exception:
                pass

        # 5. Limit pattern: "limit of sin(x)/x as x approaches 0"
        lim_match = re.search(r"limit\s+of\s+(.+?)\s+as\s+x\s+(?:approaches|->|tends\s+to)\s+([\-+0-9.a-z]+)", q_low)
        if lim_match:
            try:
                expr = parse_expr(lim_match.group(1), local_dict=local_dict, transformations=TRANSFORMATIONS)
                pt_raw = lim_match.group(2)
                pt = sp.oo if pt_raw in ("inf", "infinity", "oo") else parse_expr(pt_raw, local_dict=local_dict, transformations=TRANSFORMATIONS)
                ans = sp.limit(expr, x, pt)
                plot = _sample_function_points(expr, x)
                return {
                    "ok": True,
                    "operation": "limit",
                    "input": f"lim (x -> {pt}) [{expr}]",
                    "result": str(ans),
                    "steps": [
                        f"**Function**: `f(x) = {expr}`",
                        f"**Limit as $x \\to {pt}$**: `{ans}`",
                    ],
                    "artifacts": [plot] if plot else [],
                }
            except Exception:
                pass

        # 6. General Symbolic Expression / Equation / Calculus
        expr_str = _extract_math_expression(q_strip)
        try:
            # Equation solving if '=' is present
            if "=" in expr_str:
                lhs_s, rhs_s = expr_str.split("=", 1)
                lhs = parse_expr(lhs_s.strip(), local_dict=local_dict, transformations=TRANSFORMATIONS)
                rhs = parse_expr(rhs_s.strip(), local_dict=local_dict, transformations=TRANSFORMATIONS)
                diff_expr = sp.simplify(lhs - rhs)
                sols = sp.solve(sp.Eq(lhs, rhs), x)
                sol_formatted = [f"{s} (≈ {float(sp.N(s)):.5g})" if s.is_real and not s.is_Integer else str(s) for s in sols]
                plot = _sample_function_points(diff_expr, x) if diff_expr.has(x) else None
                return {
                    "ok": True,
                    "operation": "equation_solve",
                    "input": f"{lhs} = {rhs}",
                    "result": f"x ∈ {{ {', '.join(sol_formatted)} }}" if sol_formatted else "No closed-form real solution found",
                    "steps": [
                        f"**Standard Form**: `{diff_expr} = 0`",
                        f"**Factored Form**: `{sp.factor(diff_expr)} = 0`",
                        f"**Roots / Solutions**: `{', '.join(sol_formatted)}`",
                    ],
                    "artifacts": [plot] if plot else [],
                }

            expr = parse_expr(expr_str, local_dict=local_dict, transformations=TRANSFORMATIONS)

            if expr.has(x):
                deriv = sp.simplify(sp.diff(expr, x))
                second_deriv = sp.simplify(sp.diff(deriv, x))
                try:
                    integ = sp.integrate(expr, x)
                except Exception:
                    integ = "Non-elementary antiderivative"
                factored = sp.factor(expr)
                plot = _sample_function_points(expr, x)

                if any(k in q_low for k in ("integrate", "integral", "antiderivative")):
                    op = "indefinite_integral"
                    primary = f"{integ} + C"
                elif any(k in q_low for k in ("factor",)):
                    op = "factorization"
                    primary = str(factored)
                elif any(k in q_low for k in ("taylor", "series")):
                    op = "taylor_series"
                    primary = str(sp.series(expr, x, 0, 6))
                elif any(k in q_low for k in ("differentiate", "derivative", "derive")):
                    op = "differentiation"
                    primary = f"f'(x) = {deriv}"
                elif any(k in q_low for k in ("simplify", "expand")):
                    op = "algebraic_transform"
                    primary = str(sp.expand(expr) if "expand" in q_low else sp.simplify(expr))
                else:
                    op = "function_analysis"
                    primary = f"f(x) = {sp.simplify(expr)} | f'(x) = {deriv}"

                steps = [
                    f"**Parsed Symbolic Expression**: `f(x) = {expr}`",
                    f"**First Derivative $f'(x)$**: `{deriv}`",
                    f"**Second Derivative $f''(x)$**: `{second_deriv}`",
                    f"**Indefinite Integral $\\int f(x)\\,dx$**: `{integ} + C`",
                ]
                if str(factored) != str(expr):
                    steps.append(f"**Factored Form**: `{factored}`")

                return {
                    "ok": True,
                    "operation": op,
                    "input": f"f(x) = {expr}",
                    "result": primary,
                    "derivative": str(deriv),
                    "integral": f"{integ} + C",
                    "steps": steps,
                    "artifacts": [plot] if plot else [],
                }
            else:
                # Pure numerical evaluation
                exact = sp.simplify(expr)
                approx = float(sp.N(exact))
                res_str = str(exact) if exact.is_Integer else f"{exact} (≈ {approx:,.8g})"
                return {
                    "ok": True,
                    "operation": "numerical_evaluation",
                    "input": str(expr),
                    "result": res_str,
                    "steps": [
                        f"**Parsed Expression**: `{expr}`",
                        f"**Exact Symbolic Value**: `{exact}`",
                        f"**Decimal Approximation**: `{approx:,.10g}`",
                    ],
                    "artifacts": [],
                }
        except Exception as exc:
            return {"ok": False, "error": str(exc), "artifacts": []}
