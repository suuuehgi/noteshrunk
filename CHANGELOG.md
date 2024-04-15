# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.3] - 2024-04-15

### Changed

- Replaced `--global_palette` with `--local_palette` to make the global palette the default.

## [1.2] - 2024-04-13

### Added

- This changelog file
- Added a check for the existence of all input files so that the error is raised immediately and not halfway through processing.

### Changed

- Switch from multi-processing to multi-threading for lower overhead, resulting in lower memory consumption and faster runtime.
- The location of the temporary folder for intermediate files has been changed to the current working location, so that it can be freely selected.

## [1.1] - 2024-04-11

### Added

- Multiprocessing
  - Added `--jobs` flag
- Added `pipx` builds
- Added `--quality` flag

## [1.0] - 2024-04-10

### Added

- Initial release
