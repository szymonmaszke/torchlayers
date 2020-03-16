#!/usr/bin/env bash

repository="$(basename "$(git rev-parse --show-toplevel)")"
# Version equal to current timestamp
echo "__version__ = \"$(date +%s)\"" >"$repository"/_version.py
echo "_name = \"$repository-nightly\"" >"$repository"/_name.py
