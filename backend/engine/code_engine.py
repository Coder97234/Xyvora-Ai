import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Tuple

ALLOWED_IMPORTS = {
    "math",
    "random",
    "statistics",
    "collections",
    "itertools",
    "functools",
    "decimal",
    "fractions",
    "datetime",
    "time",
    "json",
    "re",
    "heapq",
    "bisect",
    "string",
    "numpy",
    "sympy",
}

FORBIDDEN_CALLS = {"open", "exec", "eval", "compile", "__import__", "input", "breakpoint"}


def validate_python_ast(code: str) -> Tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"SyntaxError on line {exc.lineno}: {exc.msg}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    return False, f"Import '{root}' is blocked in the Xyvora sandbox. Allowed modules: {', '.join(sorted(ALLOWED_IMPORTS))}."
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                return False, f"Import from '{root}' is blocked in the Xyvora sandbox."
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                return False, f"Built-in '{node.func.id}()' is restricted inside the isolated sandbox."
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                return False, f"Dunder attribute access '{node.attr}' is restricted in the sandbox."

    return True, "OK"


SANDBOX_PRELUDE = """\
import json as _json

_XYVORA_CHARTS = []

def emit_chart(title, labels, values, chart_type="bar"):
    _XYVORA_CHARTS.append({
        "type": "chart",
        "chart_type": chart_type,
        "title": str(title),
        "data": [{"label": str(l), "value": round(float(v), 4)} for l, v in zip(labels, values)]
    })
"""

SANDBOX_EPILOGUE = """\

if _XYVORA_CHARTS:
    print("\\n__XYVORA_ARTIFACTS__:" + _json.dumps(_XYVORA_CHARTS))
"""


class CodeEngine:
    def execute_python(self, code: str, timeout_sec: float = 6.0) -> Dict[str, Any]:
        ok, reason = validate_python_ast(code)
        if not ok:
            return {
                "ok": False,
                "stdout": "",
                "stderr": reason,
                "duration_ms": 0.0,
                "artifacts": [],
            }

        wrapped = SANDBOX_PRELUDE + "\n" + code + "\n" + SANDBOX_EPILOGUE
        t0 = time.perf_counter()

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
            tmp.write(wrapped)
            tmp_path = tmp.name

        try:
            clean_env = {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "PYTHONPATH": os.pathsep.join(p for p in sys.path if p),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            proc = subprocess.run(
                [sys.executable, "-B", tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                env=clean_env,
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            raw_out = proc.stdout or ""
            artifacts: List[Dict[str, Any]] = []

            if "__XYVORA_ARTIFACTS__:" in raw_out:
                parts = raw_out.split("__XYVORA_ARTIFACTS__:", 1)
                clean_out = parts[0].rstrip()
                try:
                    artifacts = json.loads(parts[1].strip())
                except Exception:
                    artifacts = []
            else:
                clean_out = raw_out.rstrip()

            return {
                "ok": proc.returncode == 0,
                "stdout": clean_out[:8000],
                "stderr": (proc.stderr or "").strip()[:3000],
                "duration_ms": elapsed_ms,
                "artifacts": artifacts,
            }
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "stdout": "",
                "stderr": f"Execution timed out after {timeout_sec:.1f}s (infinite loop protection).",
                "duration_ms": round(timeout_sec * 1000.0, 1),
                "artifacts": [],
            }
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def synthesize_code(self, prompt: str) -> Dict[str, Any]:
        p = prompt.lower()

        # Extract number parameter if user specified e.g. "5000 dice rolls"
        nums = [int(n) for n in re.findall(r"\b\d+\b", prompt) if 2 <= int(n) <= 200000]

        if "dice" in p or "die" in p:
            trials = nums[0] if nums else 1000
            code = f"""import random
from collections import Counter

random.seed(42)
trials = {trials}
rolls = [random.randint(1, 6) + random.randint(1, 6) for _ in range(trials)]
counts = Counter(rolls)

labels = [str(total) for total in range(2, 13)]
frequencies = [counts[total] for total in range(2, 13)]

print(f"Simulated {{trials}} two-dice rolls:")
for total in range(2, 13):
    pct = (counts[total] / trials) * 100
    print(f"  Sum {{total:2d}}: {{counts[total]:4d}} hits ({{pct:5.2f}}%)")

emit_chart(f"Two-Dice Sum Distribution (n={{trials}})", labels, frequencies, "bar")
"""
            return {"language": "python", "title": f"Monte Carlo Two-Dice Simulation (n={trials})", "code": code, "executable": True}

        if "monte carlo" in p or ("pi" in p and "simulate" in p):
            trials = nums[0] if nums else 25000
            code = f"""import random
import math

random.seed(7)
samples = {trials}
inside = 0
checkpoints = []
labels = []

step = max(1, samples // 10)
for i in range(1, samples + 1):
    x, y = random.random(), random.random()
    if x * x + y * y <= 1.0:
        inside += 1
    if i % step == 0:
        est = 4.0 * inside / i
        labels.append(f"{{i//1000}}k" if i >= 1000 else str(i))
        checkpoints.append(est)

pi_est = 4.0 * inside / samples
err = abs(math.pi - pi_est)
print(f"Samples: {{samples:,}}")
print(f"Estimated Pi : {{pi_est:.6f}}")
print(f"Actual Pi    : {{math.pi:.6f}}")
print(f"Absolute Err : {{err:.6f}}")

emit_chart("Monte Carlo Convergence Toward Pi", labels, checkpoints, "line")
"""
            return {"language": "python", "title": "Monte Carlo Pi Estimation", "code": code, "executable": True}

        if "compound interest" in p or "investment" in p:
            years = nums[0] if (nums and nums[0] <= 60) else 20
            code = f"""principal = 10000.0
annual_rate = 0.08
monthly_contrib = 300.0
years = {years}

labels = []
balances = []
balance = principal

for yr in range(1, years + 1):
    for _ in range(12):
        balance = balance * (1 + annual_rate / 12) + monthly_contrib
    labels.append(f"Yr {{yr}}")
    balances.append(round(balance, 2))

total_invested = principal + monthly_contrib * 12 * years
print(f"Initial Principal : ${{principal:,.2f}}")
print(f"Total Contributed : ${{total_invested:,.2f}}")
print(f"Final Balance     : ${{balance:,.2f}}")
print(f"Compound Growth   : ${{balance - total_invested:,.2f}}")

emit_chart(f"Compound Growth Over {{years}} Years (8% APY)", labels, balances, "area")
"""
            return {"language": "python", "title": "Compound Interest & Portfolio Growth Simulator", "code": code, "executable": True}

        if "random walk" in p or "coin" in p:
            steps = nums[0] if (nums and nums[0] <= 500) else 60
            code = f"""import random

random.seed(21)
steps = {steps}
position = 0
labels = []
positions = []

for step in range(1, steps + 1):
    position += random.choice([-1, 1])
    labels.append(str(step))
    positions.append(position)

print(f"Completed {{steps}}-step 1D symmetric random walk.")
print(f"Peak Displacement : {{max(positions)}}")
print(f"Min Displacement  : {{min(positions)}}")
print(f"Final Position    : {{position}}")

emit_chart(f"1D Random Walk Trajectory ({{steps}} Steps)", labels, positions, "line")
"""
            return {"language": "python", "title": "1D Symmetric Random Walk Simulation", "code": code, "executable": True}

        if "prime" in p:
            limit = nums[0] if (nums and nums[0] <= 50000) else 200
            code = f"""def sieve_of_eratosthenes(limit: int) -> list[int]:
    if limit < 2:
        return []
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if is_prime[p]:
            for multiple in range(p * p, limit + 1, p):
                is_prime[multiple] = False
    return [n for n, flag in enumerate(is_prime) if flag]

primes = sieve_of_eratosthenes({limit})
twin_primes = [(primes[i], primes[i + 1]) for i in range(len(primes) - 1) if primes[i + 1] - primes[i] == 2]

print(f"Found {{len(primes)}} primes <= {limit}:")
print(primes[:30], "..." if len(primes) > 30 else "")
print(f"Found {{len(twin_primes)}} twin prime pairs <= {limit}:")
print(twin_primes[:12])
"""
            return {"language": "python", "title": "Sieve of Eratosthenes & Twin Prime Finder", "code": code, "executable": True}

        if "fibonacci" in p:
            n_count = nums[0] if (nums and nums[0] <= 80) else 20
            code = f"""from functools import lru_cache

@lru_cache(maxsize=None)
def fib(n: int) -> int:
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

seq = [fib(i) for i in range({n_count})]
print(f"First {n_count} Fibonacci numbers:")
print(seq)
emit_chart("Fibonacci Growth Curve", [str(i) for i in range(min(15, {n_count}))], seq[:15], "line")
"""
            return {"language": "python", "title": "Memoized Fibonacci Sequence", "code": code, "executable": True}

        if "lru" in p and "cache" in p:
            code = """from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.store: OrderedDict[str, int] = OrderedDict()

    def get(self, key: str) -> int:
        if key not in self.store:
            return -1
        self.store.move_to_end(key)
        return self.store[key]

    def put(self, key: str, value: int) -> None:
        if key in self.store:
            self.store.move_to_end(key)
        self.store[key] = value
        if len(self.store) > self.capacity:
            evicted = self.store.popitem(last=False)
            print(f"Evicted least-recently-used entry: {evicted}")

cache = LRUCache(capacity=3)
cache.put("alpha", 10)
cache.put("beta", 20)
cache.put("gamma", 30)
print("get('alpha') ->", cache.get("alpha"))
cache.put("delta", 40)  # evicts 'beta'
print("get('beta')  ->", cache.get("beta"))
print("Active keys  ->", list(cache.store.keys()))
"""
            return {"language": "python", "title": "O(1) LRU Cache Implementation", "code": code, "executable": True}

        if "binary search" in p or "merge sort" in p or "sort" in p:
            code = """def merge_sort(items: list[int]) -> list[int]:
    if len(items) <= 1:
        return items
    mid = len(items) // 2
    left = merge_sort(items[:mid])
    right = merge_sort(items[mid:])
    merged = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged

def binary_search(sorted_arr: list[int], target: int) -> int:
    lo, hi = 0, len(sorted_arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if sorted_arr[mid] == target:
            return mid
        elif sorted_arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

sample = [42, 7, 19, 88, 3, 25, 61, 14]
ordered = merge_sort(sample)
idx = binary_search(ordered, 25)
print("Original :", sample)
print("Sorted   :", ordered)
print("Index of 25:", idx)
"""
            return {"language": "python", "title": "Merge Sort O(n log n) + Binary Search O(log n)", "code": code, "executable": True}

        if "bfs" in p or "breadth first" in p or "graph" in p or "dijkstra" in p:
            code = """from collections import deque

def shortest_path_bfs(graph: dict[str, list[str]], start: str, goal: str) -> list[str]:
    queue = deque([[start]])
    visited = {start}
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == goal:
            return path
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])
    return []

network = {
    "Earth": ["Moon", "MarsRelay"],
    "Moon": ["MarsRelay", "VenusProbe"],
    "MarsRelay": ["EuropaBase", "CeresHub"],
    "CeresHub": ["TitanStation"],
    "EuropaBase": ["TitanStation"],
}

route = shortest_path_bfs(network, "Earth", "TitanStation")
print("Optimal multi-hop route:", " -> ".join(route))
"""
            return {"language": "python", "title": "Graph Shortest-Path BFS Solver", "code": code, "executable": True}

        if "typescript" in p or "javascript" in p or "fetch" in p:
            code = """export interface RetryOptions {
  retries?: number;
  baseDelayMs?: number;
  timeoutMs?: number;
}

export async function fetchWithRetry<T>(
  url: string,
  init: RequestInit = {},
  options: RetryOptions = {}
): Promise<T> {
  const { retries = 3, baseDelayMs = 300, timeoutMs = 8000 } = options;

  for (let attempt = 0; attempt <= retries; attempt++) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(url, { ...init, signal: controller.signal });
      clearTimeout(timer);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      return (await response.json()) as T;
    } catch (err) {
      clearTimeout(timer);
      if (attempt === retries) throw err;
      const backoff = baseDelayMs * Math.pow(2, attempt);
      await new Promise((r) => setTimeout(r, backoff));
    }
  }
  throw new Error("Unreachable");
}"""
            return {"language": "typescript", "title": "TypeScript Exponential-Backoff Fetch Wrapper", "code": code, "executable": False}

        if "sql" in p:
            code = """WITH customer_totals AS (
    SELECT
        c.customer_id,
        c.full_name,
        COUNT(o.order_id) AS order_count,
        SUM(o.total_amount) AS lifetime_revenue,
        AVG(o.total_amount) AS avg_order_value
    FROM customers c
    JOIN orders o ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
      AND o.created_at >= CURRENT_DATE - INTERVAL '12 months'
    GROUP BY c.customer_id, c.full_name
)
SELECT
    customer_id,
    full_name,
    order_count,
    ROUND(lifetime_revenue, 2) AS lifetime_revenue,
    ROUND(avg_order_value, 2) AS avg_order_value,
    RANK() OVER (ORDER BY lifetime_revenue DESC) AS revenue_rank
FROM customer_totals
ORDER BY lifetime_revenue DESC
LIMIT 5;"""
            return {"language": "sql", "title": "Analytical Top-Customers SQL CTE & Window Query", "code": code, "executable": False}

        # General-purpose clean Python data & statistical toolkit fallback that actually executes!
        code = """import statistics

dataset = [14.2, 18.7, 22.1, 19.5, 27.4, 31.0, 29.8, 35.6]
mean_val = statistics.mean(dataset)
stdev_val = statistics.stdev(dataset)
z_scores = [round((x - mean_val) / stdev_val, 3) for x in dataset]

print(f"Samples  : {dataset}")
print(f"Mean     : {mean_val:.2f}")
print(f"Std Dev  : {stdev_val:.2f}")
print(f"Z-Scores : {z_scores}")

emit_chart("Sample Progression", [f"T{i+1}" for i in range(len(dataset))], dataset, "line")
"""
        return {"language": "python", "title": "Python Analytical Pipeline", "code": code, "executable": True}
