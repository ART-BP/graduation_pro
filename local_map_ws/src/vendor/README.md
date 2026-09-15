# Vendored dependencies

This directory contains the minimum ROS 1 packages required from
ANYbotics `grid_map` release `1.6.4`:

- `grid_map_core`
- `grid_map_msgs`

Upstream: <https://github.com/ANYbotics/grid_map>

The unmodified upstream license is stored in `GRID_MAP_LICENSE`.
These packages are kept in the workspace because the current development
machine does not have permission to install the corresponding Debian packages.
Only the circular-grid container and ROS message definition are retained.
The small message converter used by this project is implemented locally, so
OpenCV and the full `grid_map_ros` package are not required.
