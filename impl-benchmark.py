#!/usr/bin/env python3

# impl-benchmark.py -- benchmark various implementations
# Copyright (C) 2026 Alexios Zavras
# SPDX-License-Identifier: GPL-3.0-or-later

# This script benchmarks multiple command-line utilities by running them
# multiple times in a randomized order, measuring execution time and verifying
# output consistency. It then reports statistics including average, min, max,
# median, and standard deviation of execution times, as well as relative performance.
# example runs:
"""
# Benchmark different sha1sum implementations
./benchmark.py -r 50 -a largefile.txt \
    "system_sha1:sha1sum" \
    "openssl:openssl sha1" \
    "python:python3 -c import hashlib,sys; print(hashlib.sha1(open(sys.argv[1],'rb').read()).hexdigest())"
# Benchmark different compression tools
./benchmark.py -r 20 -a testfile.txt \
    "gzip:gzip -c" \
    "bzip2:bzip2 -c" \
    "xz:xz -c"
# Use default /usr/bin/time
./benchmark.py -m -r 20 "cmd1:echo hello" "cmd2:printf hello"
# Use custom time executable
./benchmark.py -m -T /usr/local/bin/time -r 20 "cmd1:echo hello" "cmd2:printf hello"
# Use gtime on macOS (GNU time via Homebrew)
./benchmark.py -m -T /usr/local/bin/gtime -r 20 "cmd1:gzip -c" "cmd2:bzip2 -c"
# Export to CSV with reproducible seed
./benchmark.py -r 30 -s 42 --csv results.csv "cmd1:sleep 0.1" "cmd2:sleep 0.2"
"""

import argparse
import csv
import json
import pathlib
import random
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict


def parse_time_output(stderr):
    """Parse /usr/bin/time -v output to extract resource metrics."""
    metrics = {}
    # Patterns for various metrics
    patterns = {
        "user_time": r"User time \(seconds\): ([\d.]+)",
        "system_time": r"System time \(seconds\): ([\d.]+)",
        "elapsed_time": r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:\.]+)",
        "cpu_percent": r"Percent of CPU this job got: (\d+)%",
        "max_rss_kb": r"Maximum resident set size \(kbytes\): (\d+)",
        "major_faults": r"Major \(requiring I/O\) page faults: (\d+)",
        "minor_faults": r"Minor \(reclaiming a frame\) page faults: (\d+)",
        "voluntary_switches": r"Voluntary context switches: (\d+)",
        "involuntary_switches": r"Involuntary context switches: (\d+)",
        "fs_inputs": r"File system inputs: (\d+)",
        "fs_outputs": r"File system outputs: (\d+)",
        "page_reclaims": r"Page reclaims \(soft page faults\): (\d+)",
        "context_switches": r"Context switches: (\d+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, stderr)
        if match:
            value = match.group(1)
            # Convert elapsed time to seconds if needed
            if key == "elapsed_time":
                parts = value.split(":")
                if len(parts) == 2:  # m:ss
                    metrics[key] = float(parts[0]) * 60 + float(parts[1])
                elif len(parts) == 3:  # h:mm:ss
                    metrics[key] = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            elif key == "cpu_percent":
                metrics[key] = int(value)
            else:
                metrics[key] = float(value)
    return metrics


def run_command(cmd, cmd_arg=None, timeout=None, collect_metrics=False, time_cmd=None):
    """Run a command and measure its execution time and optionally resource usage."""
    start_time = time.perf_counter()
    try:
        # Wrap command with time executable if metrics collection is enabled
        if collect_metrics and time_cmd:
            timing_cmd = [time_cmd, "-v"] + cmd
            if cmd_arg:
                timing_cmd.append(cmd_arg)
            result = subprocess.run(timing_cmd, check=False, capture_output=True, text=True, timeout=timeout)
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            # time command writes to stderr
            metrics = parse_time_output(result.stderr)
            return {
                "time": execution_time,
                "stdout": result.stdout.strip(),
                "metrics": metrics,
                "success": result.returncode == 0,
                "error": None if result.returncode == 0 else f"Exit code: {result.returncode}",
            }
        # Standard execution without detailed metrics
        if cmd_arg:
            result = subprocess.run(cmd + [cmd_arg], capture_output=True, text=True, check=True, timeout=timeout)
        else:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        return {"time": execution_time, "stdout": result.stdout.strip(), "success": True}
    except subprocess.TimeoutExpired:
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        return {"time": execution_time, "stdout": "", "success": False, "error": "Command timed out"}
    except subprocess.CalledProcessError as e:
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        return {"time": execution_time, "stdout": "", "success": False, "error": str(e)}


def detect_outliers(times):
    """Detect outliers using IQR method. Returns list of booleans indicating outliers."""
    if len(times) < 4:
        return [False] * len(times)
    sorted_times = sorted(times)
    q1 = statistics.quantiles(sorted_times, n=4)[0]
    q3 = statistics.quantiles(sorted_times, n=4)[2]
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return [t < lower_bound or t > upper_bound for t in times]


def calculate_percentiles(times):
    """Calculate common percentiles."""
    if not times:
        return {}
    sorted_times = sorted(times)
    if len(sorted_times) == 1:
        return {"p50": sorted_times[0], "p90": sorted_times[0], "p95": sorted_times[0], "p99": sorted_times[0]}
    percentiles = statistics.quantiles(sorted_times, n=100)
    return {
        "p50": percentiles[49],  # median
        "p90": percentiles[89],
        "p95": percentiles[94],
        "p99": percentiles[98] if len(percentiles) > 98 else percentiles[-1],
    }


def benchmark_commands(
    commands, total_runs, cmd_arg=None, verify_output=True, warmup=0, timeout=None, collect_metrics=False, time_cmd=None, seed=None
):
    """Benchmark multiple commands with random execution order."""
    # Check if time command is available when metrics collection is requested
    if collect_metrics:
        if not time_cmd or not pathlib.Path(time_cmd).exists():
            print(f"Warning: time command '{time_cmd}' not found. Detailed metrics will not be collected.")
            collect_metrics = False
            time_cmd = None
    # Warmup phase
    if warmup > 0:
        print(f"Running {warmup} warmup iterations per command...")
        for name, cmd in commands.items():
            for _ in range(warmup):
                run_command(cmd, cmd_arg, timeout, collect_metrics=False, time_cmd=None)
        print("Warmup complete!\n")
    # Create a list of all runs to execute
    runs = []
    runs_per_cmd = total_runs // len(commands)
    remainder = total_runs % len(commands)
    for i, (name, cmd) in enumerate(commands.items()):
        # Distribute remainder runs among first few commands
        cmd_runs = runs_per_cmd + (1 if i < remainder else 0)
        runs.extend([(name, cmd)] * cmd_runs)
    # Randomize execution order with optional seed for reproducibility
    if seed is not None:
        random.seed(seed)
        print(f"Using random seed: {seed}")
    random.shuffle(runs)
    # Store results with individual run data
    results = defaultdict(lambda: {"times": [], "metrics": defaultdict(list), "runs": []})
    reference_output = None
    print(f"Running {len(runs)} total benchmarks...")
    for i, (name, cmd) in enumerate(runs, 1):
        print(f"Progress: {i}/{len(runs)} ({100 * i // len(runs)}%) - Running {name}...".ljust(70), end="\r")
        result = run_command(cmd, cmd_arg, timeout, collect_metrics, time_cmd)
        if not result["success"]:
            print(f"\nError running {name}: {result['error']}")
            continue
        results[name]["times"].append(result["time"])
        # Store individual run data for CSV export
        run_data = {"run_number": len(results[name]["runs"]) + 1, "time": result["time"]}
        # Store metrics if available
        if "metrics" in result:
            for metric_name, metric_value in result["metrics"].items():
                results[name]["metrics"][metric_name].append(metric_value)
                run_data[metric_name] = metric_value
        results[name]["runs"].append(run_data)
        # Verify output consistency (optional)
        if verify_output:
            if reference_output is None:
                reference_output = result["stdout"]
            elif result["stdout"] != reference_output:
                print(f"\nWarning: {name} produced different output!")
    print("\nBenchmark complete!".ljust(70))
    return results


def print_statistics(results, show_outliers=False, show_metrics=False):
    """Print benchmark statistics."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    # Sort by average time
    sorted_results = sorted(results.items(), key=lambda x: statistics.mean(x[1]["times"]) if x[1]["times"] else float("inf"))
    for name, data in sorted_results:
        times = data["times"]
        if not times:
            print(f"{name}: No successful runs")
            continue
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        median_time = statistics.median(times)
        print(f"\n{name}:")
        print(f"  Runs:    {len(times)}")
        print(f"  Average: {avg_time:.4f}s")
        print(f"  Median:  {median_time:.4f}s")
        print(f"  Min:     {min_time:.4f}s")
        print(f"  Max:     {max_time:.4f}s")
        if len(times) > 1:
            stdev = statistics.stdev(times)
            cv = (stdev / avg_time) * 100  # Coefficient of variation
            print(f"  Std Dev: {stdev:.4f}s ({cv:.1f}%)")
            # Percentiles
            percentiles = calculate_percentiles(times)
            print(f"  P90:     {percentiles['p90']:.4f}s")
            print(f"  P95:     {percentiles['p95']:.4f}s")
            # Outlier detection
            if show_outliers:
                outliers = detect_outliers(times)
                outlier_count = sum(outliers)
                if outlier_count > 0:
                    print(f"  Outliers: {outlier_count}/{len(times)}")
        # Show resource metrics if available
        if show_metrics and data["metrics"]:
            print("\n  Resource Metrics (averages):")
            metrics = data["metrics"]
            if "user_time" in metrics:
                print(f"    CPU User Time:   {statistics.mean(metrics['user_time']):.4f}s")
            if "system_time" in metrics:
                print(f"    CPU System Time: {statistics.mean(metrics['system_time']):.4f}s")
            if "cpu_percent" in metrics:
                print(f"    CPU Usage:       {statistics.mean(metrics['cpu_percent']):.1f}%")
            if "max_rss_kb" in metrics:
                avg_mem_mb = statistics.mean(metrics["max_rss_kb"]) / 1024
                print(f"    Max Memory:      {avg_mem_mb:.2f} MB")
            if "major_faults" in metrics:
                print(f"    Major Faults:    {statistics.mean(metrics['major_faults']):.1f}")
            if "minor_faults" in metrics:
                print(f"    Minor Faults:    {statistics.mean(metrics['minor_faults']):.1f}")
            if "voluntary_switches" in metrics:
                print(f"    Vol. Ctx Sw.:    {statistics.mean(metrics['voluntary_switches']):.1f}")
            if "involuntary_switches" in metrics:
                print(f"    Invol. Ctx Sw.:  {statistics.mean(metrics['involuntary_switches']):.1f}")
            if "fs_inputs" in metrics:
                print(f"    FS Inputs:       {statistics.mean(metrics['fs_inputs']):.1f}")
            if "fs_outputs" in metrics:
                print(f"    FS Outputs:      {statistics.mean(metrics['fs_outputs']):.1f}")
    # Print relative performance
    if len(sorted_results) > 1:
        print("\n" + "-" * 70)
        print("RELATIVE PERFORMANCE")
        print("-" * 70)
        fastest_name = sorted_results[0][0]
        fastest_avg = statistics.mean(sorted_results[0][1]["times"])
        for name, data in sorted_results:
            times = data["times"]
            if times:
                avg_time = statistics.mean(times)
                if name == fastest_name:
                    print(f"{name}: baseline (fastest)")
                else:
                    ratio = avg_time / fastest_avg
                    percent_slower = (ratio - 1) * 100
                    print(f"{name}: {ratio:.2f}x ({percent_slower:.1f}% slower)")


def export_results(results, filename="benchmark_results.json"):
    """Export results to JSON file."""
    export_data = {}
    for name, data in results.items():
        times = data["times"]
        if times:
            export_data[name] = {
                "times": times,
                "average": statistics.mean(times),
                "median": statistics.median(times),
                "min": min(times),
                "max": max(times),
                "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
                "runs": len(times),
            }
            if len(times) > 1:
                percentiles = calculate_percentiles(times)
                export_data[name]["percentiles"] = percentiles
            # Export metrics if available
            if data["metrics"]:
                export_data[name]["resource_metrics"] = {}
                for metric_name, metric_values in data["metrics"].items():
                    export_data[name]["resource_metrics"][metric_name] = {
                        "average": statistics.mean(metric_values),
                        "min": min(metric_values),
                        "max": max(metric_values),
                        "values": metric_values,
                    }
    with pathlib.Path(filename).open("w") as f:
        json.dump(export_data, f, indent=2)
    print(f"\nResults exported to {filename}")


def export_results_csv(results, filename="benchmark_results.csv"):
    """Export detailed results to CSV file including all metrics."""
    # Collect all possible metric names across all commands
    all_metrics = set()
    for data in results.values():
        if data["runs"]:
            for run in data["runs"]:
                all_metrics.update(run.keys())
    # Remove 'run_number' and 'time' as they'll be first columns
    all_metrics.discard("run_number")
    all_metrics.discard("time")
    metric_names = sorted(all_metrics)
    # Build header
    header = ["command", "run_number", "time_seconds"]
    # Add metric columns with units/descriptions
    metric_headers = {
        "user_time": "cpu_user_time_s",
        "system_time": "cpu_system_time_s",
        "elapsed_time": "elapsed_time_s",
        "cpu_percent": "cpu_percent",
        "max_rss_kb": "max_memory_kb",
        "major_faults": "major_page_faults",
        "minor_faults": "minor_page_faults",
        "voluntary_switches": "voluntary_context_switches",
        "involuntary_switches": "involuntary_context_switches",
        "fs_inputs": "filesystem_inputs",
        "fs_outputs": "filesystem_outputs",
        "page_reclaims": "page_reclaims",
        "context_switches": "context_switches",
    }
    for metric in metric_names:
        header.append(metric_headers.get(metric, metric))
    with pathlib.Path(filename).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        # Write data rows
        for name, data in sorted(results.items()):
            for run in data["runs"]:
                row = [name, run["run_number"], f"{run['time']:.6f}"]
                # Add metric values in the same order as headers
                for metric in metric_names:
                    value = run.get(metric, "")
                    if value != "":
                        row.append(f"{value:.6f}" if isinstance(value, float) else str(value))
                    else:
                        row.append("")
                writer.writerow(row)
    print(f"\nDetailed results exported to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark command-line utilities")
    parser.add_argument("-r", "--runs", type=int, default=30, help="Total number of runs (default: 30)")
    parser.add_argument("-a", "--arg", type=str, help="Argument to pass to commands")
    parser.add_argument("--no-verify", action="store_true", help="Skip output verification")
    parser.add_argument("-w", "--warmup", type=int, default=3, help="Number of warmup runs per command (default: 3)")
    parser.add_argument("-t", "--timeout", type=float, help="Command timeout in seconds")
    parser.add_argument("-o", "--output", type=str, help="Export results to JSON file")
    parser.add_argument("--csv", type=str, help="Export detailed results to CSV file")
    parser.add_argument("--show-outliers", action="store_true", help="Show outlier detection in statistics")
    parser.add_argument("-m", "--metrics", action="store_true", help="Collect detailed resource metrics using time command")
    parser.add_argument("-T", "--timecmd", type=str, default="/usr/local/bin/time", help="Path to time command executable (default: /usr/bin/time)")
    parser.add_argument("-s", "--seed", type=int, help="Random seed for reproducible test ordering")
    parser.add_argument("commands", nargs="+", help="Commands to benchmark (format: name:command)")
    args = parser.parse_args()
    # Parse commands
    commands = {}
    for cmd_spec in args.commands:
        if ":" not in cmd_spec:
            print(f"Error: Command spec must be in format 'name:command', got: {cmd_spec}")
            sys.exit(1)
        name, cmd_str = cmd_spec.split(":", 1)
        commands[name] = cmd_str.split()
    print(f"Commands to benchmark: {list(commands.keys())}")
    print(f"Total runs: {args.runs}")
    if args.warmup > 0:
        print(f"Warmup runs: {args.warmup} per command")
    if args.arg:
        print(f"Command argument: {args.arg}")
    if args.timeout:
        print(f"Timeout: {args.timeout}s")
    if args.metrics:
        print(f"Collecting detailed resource metrics using: {args.timecmd}")
    if args.seed is not None:
        print(f"Random seed: {args.seed}")
    # Run benchmarks
    results = benchmark_commands(
        commands,
        args.runs,
        args.arg,
        not args.no_verify,
        args.warmup,
        args.timeout,
        args.metrics,
        args.timecmd if args.metrics else None,
        args.seed,
    )
    # Print results
    print_statistics(results, args.show_outliers, args.metrics)
    # Export if requested
    if args.output:
        export_results(results, args.output)
    if args.csv:
        export_results_csv(results, args.csv)


if __name__ == "__main__":
    main()

