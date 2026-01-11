# impl-benchmark

A simple Python script to benchmark various implementations (usually of the same thing).


## Features

With this script, one is able to:

- [x] specify different implementation commands on the command-line
- [x] specify number of runs
- [x] report execution time of each implementation
- [x] report statistics on the distribution of execution times (min, max, avg, median)
- [x] report advanced statistics on the distribution of execution times (outliers)
- [x] give short names to implmentations (in the form of `name:command`)
- [x] specify the common argument to all implementations
- [x] use arbitrary command options to each implementation
- [x] verify that the output of each implementation run is consistent
- [x] use `/usr/bin/time` to get more metrics (like max memory used)
- [x] specify alternate executable for `time`
- [x] perform warm-up runs
- [x] specify the number of warm-up runs
- [x] export results to JSON
- [x] export results to CSV

## Usage

```sh
$ ./impl-benchmark.py -h
usage: impl-benchmark.py [-h] [-r RUNS] [-a ARG] [--no-verify] [-w WARMUP]
                         [-t TIMEOUT] [-o OUTPUT] [--csv CSV]
                         [--show-outliers] [-m] [-T TIMECMD] [-s SEED]
                         commands [commands ...]

Benchmark command-line utilities

positional arguments:
  commands              Commands to benchmark (format: name:command)

options:
  -h, --help            show this help message and exit
  -r, --runs RUNS       Total number of runs (default: 30)
  -a, --arg ARG         Argument to pass to commands
  --no-verify           Skip output verification
  -w, --warmup WARMUP   Number of warmup runs per command (default: 3)
  -t, --timeout TIMEOUT
                        Command timeout in seconds
  -o, --output OUTPUT   Export results to JSON file
  --csv CSV             Export detailed results to CSV file
  --show-outliers       Show outlier detection in statistics
  -m, --metrics         Collect detailed resource metrics using time command
  -T, --timecmd TIMECMD
                        Path to time command executable (default:
                        /usr/bin/time)
  -s, --seed SEED       Random seed for reproducible test ordering
```


## Example calls

Benchmark different sha1sum implementations
```sh
./benchmark.py -r 50 -a largefile.txt \
    "system_sha1:sha1sum" \
    "openssl:openssl sha1" \
    "python:python3 -c import hashlib,sys; print(hashlib.sha1(open(sys.argv[1],'rb').read()).hexdigest())"
```

Benchmark different compression tools
```sh
./benchmark.py -r 20 -a testfile.txt \
    "gzip:gzip -c" \
    "bzip2:bzip2 -c" \
    "xz:xz -c"
```

Use default /usr/bin/time
```sh
./benchmark.py -m -r 20 "cmd1:echo hello" "cmd2:printf hello"
```

Use custom time executable
```sh
./benchmark.py -m -T /usr/local/bin/time -r 20 "cmd1:echo hello" "cmd2:printf hello"
```

Export to CSV with reproducible seed
```sh
./benchmark.py -r 30 -s 42 --csv results.csv "cmd1:sleep 0.1" "cmd2:sleep 0.2"
```


## License

The script is licensed under `GPL-3.0-or-later`.

