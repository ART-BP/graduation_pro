"""Static triangle-mesh terrain bank used by the native Isaac Lab ray caster.

The bank contains every terrain family at several difficulty levels. Episodes
select a tile by moving their world origin instead of modifying geometry after
the ray caster has cached its Warp mesh.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporter, TerrainImporterCfg
from isaaclab.utils import configclass

from go2w_terrain_planner.mapping.terrain_truth_model import TERRAIN_NAMES


def _box(trimesh, size, center):
    transform = np.eye(4)
    transform[:3, 3] = np.asarray(center, dtype=np.float64)
    return trimesh.creation.box(extents=size, transform=transform)


def _sloped_block(
    trimesh,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_minimum: float,
    z_at_x_min: float,
    z_at_x_max: float,
):
    """Create a closed wedge whose upper surface varies linearly along x."""
    vertices = np.asarray(
        [
            [x_min, y_min, z_minimum],
            [x_max, y_min, z_minimum],
            [x_max, y_max, z_minimum],
            [x_min, y_max, z_minimum],
            [x_min, y_min, z_at_x_min],
            [x_max, y_min, z_at_x_max],
            [x_max, y_max, z_at_x_max],
            [x_min, y_max, z_at_x_min],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _surface_patch(trimesh, x_axis, y_axis, heights):
    """Create a two-sided height surface without an artificial plane below it."""
    grid_x, grid_y = np.meshgrid(x_axis, y_axis, indexing="ij")
    vertices = np.stack((grid_x, grid_y, heights), axis=-1).reshape(-1, 3)
    width = y_axis.size
    faces = []
    for row in range(x_axis.size - 1):
        for column in range(y_axis.size - 1):
            first = row * width + column
            second = first + width
            faces.append((first, second, first + 1))
            faces.append((first + 1, second, second + 1))
    return trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )


def _ground_panels_around_rectangle(
    trimesh,
    half_size: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    bottom_z: float,
):
    """Create four ground solids while leaving the central rectangle open."""
    thickness = -bottom_z
    center_z = 0.5 * bottom_z
    panels = []

    def append_box(x0, x1, y0, y1):
        if x1 - x0 > 1.0e-4 and y1 - y0 > 1.0e-4:
            panels.append(
                _box(
                    trimesh,
                    (x1 - x0, y1 - y0, thickness),
                    (0.5 * (x0 + x1), 0.5 * (y0 + y1), center_z),
                )
            )

    append_box(-half_size, x_min, -half_size, half_size)
    append_box(x_max, half_size, -half_size, half_size)
    append_box(x_min, x_max, -half_size, y_min)
    append_box(x_min, x_max, y_max, half_size)
    return panels


@configclass
class Go2WTerrainBankCfg(TerrainGeneratorCfg):
    """Configuration for the deterministic course-aligned terrain mesh bank."""

    class_type: type = None
    size: tuple[float, float] = (56.0, 56.0)
    num_rows: int = 17
    num_cols: int = 8 * len(TERRAIN_NAMES)
    sub_terrains: dict = {}
    curriculum: bool = True
    border_width: float = 0.0
    color_scheme: str = "height"
    variants_per_type: int = 8
    surface_resolution_m: float = 0.05
    ramp_slope_range: tuple[float, float] = (0.05, 0.40)
    step_height_range_m: tuple[float, float] = (0.05, 0.35)
    step_width_range_m: tuple[float, float] = (0.25, 0.60)
    rough_amplitude_range_m: tuple[float, float] = (0.01, 0.12)
    pit_depth_range_m: tuple[float, float] = (0.05, 0.40)
    pit_half_width_range_m: tuple[float, float] = (0.25, 0.60)
    obstacle_height_range_m: tuple[float, float] = (0.55, 2.50)
    low_obstacle_height_range_m: tuple[float, float] = (0.06, 0.30)
    barrier_half_width_range_m: tuple[float, float] = (0.40, 2.00)
    challenge_barrier_width_scale: float = 0.55


class Go2WTerrainBank:
    """Build one static mesh and retain exact geometry metadata for every tile."""

    latest_metadata: ClassVar[dict[str, np.ndarray] | None] = None

    def __init__(self, cfg: Go2WTerrainBankCfg, device: str = "cpu") -> None:
        del device
        import trimesh

        if cfg.num_rows < 2:
            raise ValueError("地形库至少需要两个难度等级")
        if cfg.variants_per_type <= 0:
            raise ValueError("每类地形的几何变体数量必须大于0")
        expected_columns = len(TERRAIN_NAMES) * cfg.variants_per_type
        if cfg.num_cols != expected_columns:
            raise ValueError(
                f"地形库列数必须为{expected_columns}，实际为{cfg.num_cols}"
            )
        if cfg.size[0] <= 0.0 or cfg.size[1] <= 0.0:
            raise ValueError("地形块尺寸必须大于0")

        shape = (cfg.num_rows, cfg.num_cols)
        metadata = {
            "terrain_type": np.zeros(shape, dtype=np.int64),
            "difficulty": np.zeros(shape, dtype=np.float32),
            "amplitude": np.zeros(shape, dtype=np.float32),
            "feature_x": np.zeros(shape, dtype=np.float32),
            "feature_y": np.zeros(shape, dtype=np.float32),
            "feature_yaw": np.zeros(shape, dtype=np.float32),
            "route_yaw": np.zeros(shape, dtype=np.float32),
            "feature_width": np.zeros(shape, dtype=np.float32),
            "barrier_half_width": np.zeros(shape, dtype=np.float32),
            "traversal_direction": np.ones(shape, dtype=np.float32),
        }
        origins = np.zeros((cfg.num_rows, cfg.num_cols, 3), dtype=np.float32)
        meshes = []
        total_x = cfg.num_rows * cfg.size[0]
        total_y = cfg.num_cols * cfg.size[1]

        for row in range(cfg.num_rows):
            difficulty = row / float(cfg.num_rows - 1)
            for column in range(cfg.num_cols):
                terrain_index = column // cfg.variants_per_type
                variant = column % cfg.variants_per_type
                center_x = (row + 0.5) * cfg.size[0] - 0.5 * total_x
                center_y = (column + 0.5) * cfg.size[1] - 0.5 * total_y
                tile_mesh, parameters = self._make_tile(
                    trimesh,
                    cfg,
                    terrain_index,
                    difficulty,
                    variant,
                )
                tile_mesh.apply_translation((center_x, center_y, 0.0))
                meshes.append(tile_mesh)
                origins[row, column] = (center_x, center_y, 0.0)
                metadata["terrain_type"][row, column] = terrain_index
                metadata["difficulty"][row, column] = difficulty
                for name, value in parameters.items():
                    metadata[name][row, column] = value

        self.terrain_mesh = trimesh.util.concatenate(meshes)
        self.terrain_origins = origins
        self.flat_patches = {}
        self.metadata = metadata
        Go2WTerrainBank.latest_metadata = metadata

    @staticmethod
    def _interpolate(bounds, difficulty: float) -> float:
        return float(bounds[0] + difficulty * (bounds[1] - bounds[0]))

    @staticmethod
    def _permuted_fraction(variant: int, count: int, multiplier: int) -> float:
        """Return a deterministic low-correlation fraction for one variant axis."""
        if count <= 1:
            return 0.5
        return float((variant * multiplier) % count) / float(count - 1)

    @classmethod
    def _variant_layout(cls, cfg, terrain_name: str, variant: int):
        """Diversify position and width without changing curriculum difficulty."""
        count = int(cfg.variants_per_type)
        forward_fraction = cls._permuted_fraction(variant, count, 3)
        lateral_fraction = cls._permuted_fraction(variant, count, 5)
        width_fraction = cls._permuted_fraction(variant, count, 7)

        # Early slope/roughness goals can be as close as 1.5 m, whereas later
        # obstacle stages allow the feature to be placed farther ahead.
        if terrain_name in ("ramp", "rough"):
            forward_bounds = (0.60, 1.25)
        elif terrain_name in ("step", "pit", "low_obstacle"):
            forward_bounds = (0.75, 1.95)
        else:
            forward_bounds = (0.85, 2.35)
        feature_x = forward_bounds[0] + forward_fraction * (
            forward_bounds[1] - forward_bounds[0]
        )

        if terrain_name in ("rough", "pit"):
            maximum_lateral = 0.45
        elif terrain_name in (
            "wall",
            "pillar",
            "mixed",
            "multi_route",
            "low_obstacle",
        ):
            maximum_lateral = 0.65
        else:
            # Flat, ramp, step and stairs span the tile width. Advertising a
            # lateral feature offset to the privileged critic would be false.
            maximum_lateral = 0.0
        feature_y = maximum_lateral * (2.0 * lateral_fraction - 1.0)
        width_scale = 0.80 + 0.40 * width_fraction
        return feature_x, feature_y, width_scale

    def _make_tile(self, trimesh, cfg, terrain_index, difficulty, variant):
        terrain_name = TERRAIN_NAMES[terrain_index]
        half_x = 0.5 * cfg.size[0] - 0.01
        half_y = 0.5 * cfg.size[1] - 0.01
        feature_x, feature_y, width_scale = self._variant_layout(
            cfg, terrain_name, variant
        )
        step_width = min(
            cfg.step_width_range_m[1],
            max(
                cfg.step_width_range_m[0],
                self._interpolate(cfg.step_width_range_m, difficulty)
                * width_scale,
            ),
        )
        barrier_half_width = self._interpolate(
            cfg.barrier_half_width_range_m,
            difficulty,
        ) * (
            cfg.challenge_barrier_width_scale
            + difficulty * (1.0 - cfg.challenge_barrier_width_scale)
        )
        barrier_half_width = min(
            cfg.barrier_half_width_range_m[1],
            barrier_half_width * width_scale,
        )
        traversal_direction = (
            -1.0
            if terrain_name == "stairs" and variant % 2 == 1
            else 1.0
        )
        # Localized features are deliberately shifted laterally.  Point the
        # task route through the physical feature centre so every variant is a
        # real planning/traversal problem instead of an accidental bypass.
        route_yaw = (
            float(np.arctan2(feature_y, feature_x))
            if terrain_name
            in (
                "rough",
                "pit",
                "wall",
                "pillar",
                "mixed",
                "multi_route",
                "low_obstacle",
            )
            else 0.0
        )
        amplitude = 0.0
        meshes = []

        def add_flat_base():
            meshes.append(
                _box(
                    trimesh,
                    (2.0 * half_x, 2.0 * half_y, 0.20),
                    (0.0, 0.0, -0.10),
                )
            )

        if terrain_name == "pit":
            amplitude = self._interpolate(cfg.pit_depth_range_m, difficulty)
            feature_width = min(
                cfg.pit_half_width_range_m[1],
                max(
                    cfg.pit_half_width_range_m[0],
                    self._interpolate(
                        cfg.pit_half_width_range_m,
                        difficulty,
                    )
                    * width_scale,
                ),
            )
            x_min = feature_x - feature_width
            x_max = feature_x + feature_width
            y_min = feature_y - 1.2 * feature_width
            y_max = feature_y + 1.2 * feature_width
            meshes.extend(
                _ground_panels_around_rectangle(
                    trimesh,
                    min(half_x, half_y),
                    x_min,
                    x_max,
                    y_min,
                    y_max,
                    -amplitude - 0.10,
                )
            )
            meshes.append(
                _box(
                    trimesh,
                    (x_max - x_min, y_max - y_min, 0.10),
                    (feature_x, feature_y, -amplitude - 0.05),
                )
            )
        elif terrain_name == "rough":
            amplitude = self._interpolate(cfg.rough_amplitude_range_m, difficulty)
            feature_width = step_width
            x_min, x_max = feature_x - 1.25, feature_x + 1.25
            y_min, y_max = feature_y - 1.20, feature_y + 1.20
            meshes.extend(
                _ground_panels_around_rectangle(
                    trimesh,
                    min(half_x, half_y),
                    x_min,
                    x_max,
                    y_min,
                    y_max,
                    -0.20,
                )
            )
            x_axis = np.arange(x_min, x_max + 0.5 * cfg.surface_resolution_m, cfg.surface_resolution_m)
            y_axis = np.arange(y_min, y_max + 0.5 * cfg.surface_resolution_m, cfg.surface_resolution_m)
            along = x_axis[:, None] - feature_x
            cross = y_axis[None, :] - feature_y
            heights = amplitude * np.sin(6.0 * along) * np.cos(5.0 * cross)
            meshes.append(_surface_patch(trimesh, x_axis, y_axis, heights))
        else:
            add_flat_base()
            feature_width = step_width
            if terrain_name == "ramp":
                amplitude = self._interpolate(cfg.ramp_slope_range, difficulty)
                x_start = feature_x
                x_end = feature_x + 1.5
                final_height = 1.5 * amplitude
                meshes.append(
                    _sloped_block(
                        trimesh, x_start, x_end, -half_y, half_y,
                        -0.10, 0.0, final_height,
                    )
                )
                meshes.append(
                    _box(
                        trimesh,
                        (half_x - x_end, 2.0 * half_y, final_height + 0.10),
                        (0.5 * (x_end + half_x), 0.0, 0.5 * (final_height - 0.10)),
                    )
                )
            elif terrain_name == "step":
                amplitude = self._interpolate(cfg.step_height_range_m, difficulty)
                meshes.append(
                    _box(
                        trimesh,
                        (half_x - feature_x, 2.0 * half_y, amplitude + 0.10),
                        (0.5 * (feature_x + half_x), 0.0, 0.5 * (amplitude - 0.10)),
                    )
                )
            elif terrain_name == "stairs":
                amplitude = self._interpolate(cfg.step_height_range_m, difficulty)
                stair_height = 0.45 * amplitude
                if traversal_direction > 0.0:
                    for count in range(1, 5):
                        x0 = feature_x + (count - 1) * step_width
                        x1 = feature_x + count * step_width if count < 4 else half_x
                        meshes.append(
                            _box(
                                trimesh,
                                (x1 - x0, 2.0 * half_y, count * stair_height + 0.10),
                                (0.5 * (x0 + x1), 0.0, 0.5 * (count * stair_height - 0.10)),
                            )
                        )
                else:
                    # Start on the upper platform and descend in +x.  The
                    # previous implementation placed the lower treads behind
                    # the robot, so odd variants looked flat along the goal.
                    top_height = 4.0 * stair_height
                    meshes.append(
                        _box(
                            trimesh,
                            (feature_x + half_x, 2.0 * half_y, top_height + 0.10),
                            (0.5 * (feature_x - half_x), 0.0, 0.5 * (top_height - 0.10)),
                        )
                    )
                    for count in range(1, 4):
                        x0 = feature_x + (count - 1) * step_width
                        x1 = feature_x + count * step_width
                        tread_height = (4 - count) * stair_height
                        meshes.append(
                            _box(
                                trimesh,
                                (x1 - x0, 2.0 * half_y, tread_height + 0.10),
                                (0.5 * (x0 + x1), 0.0, 0.5 * (tread_height - 0.10)),
                            )
                        )
            elif terrain_name == "wall":
                amplitude = self._interpolate(cfg.obstacle_height_range_m, difficulty)
                meshes.append(
                    _box(
                        trimesh,
                        (0.20, 2.0 * barrier_half_width, amplitude),
                        (feature_x, feature_y, 0.5 * amplitude),
                    )
                )
            elif terrain_name == "pillar":
                amplitude = self._interpolate(cfg.obstacle_height_range_m, difficulty)
                cylinder = trimesh.creation.cylinder(
                    radius=feature_width,
                    height=amplitude,
                    sections=32,
                )
                cylinder.apply_translation((feature_x, feature_y, 0.5 * amplitude))
                meshes.append(cylinder)
            elif terrain_name == "mixed":
                amplitude = self._interpolate(cfg.step_height_range_m, difficulty)
                x_start = feature_x - 0.75
                x_end = feature_x + 0.75
                ramp_height = 0.45 * amplitude
                meshes.append(
                    _sloped_block(
                        trimesh, x_start, x_end, -half_y, half_y,
                        -0.10, 0.0, ramp_height,
                    )
                )
                meshes.append(
                    _box(
                        trimesh,
                        (half_x - x_end, 2.0 * half_y, ramp_height + 0.10),
                        (0.5 * (x_end + half_x), 0.0, 0.5 * (ramp_height - 0.10)),
                    )
                )
                meshes.append(
                    _sloped_block(
                        trimesh, feature_x, x_end, -half_y, feature_y,
                        -0.10, amplitude + 0.225 * amplitude, amplitude + ramp_height,
                    )
                )
                meshes.append(
                    _box(
                        trimesh,
                        (half_x - x_end, feature_y + half_y, amplitude + ramp_height + 0.10),
                        (0.5 * (x_end + half_x), 0.5 * (feature_y - half_y),
                         0.5 * (amplitude + ramp_height - 0.10)),
                    )
                )
                wall_x = feature_x + 0.8
                wall_height = 0.8 + amplitude
                meshes.append(
                    _box(
                        trimesh,
                        (0.24, barrier_half_width, wall_height),
                        (wall_x, feature_y + 0.5 * barrier_half_width,
                         ramp_height + 0.5 * wall_height),
                    )
                )
            elif terrain_name == "multi_route":
                amplitude = self._interpolate(cfg.step_height_range_m, difficulty)
                meshes.append(
                    _box(
                        trimesh,
                        (0.36, 2.0 * barrier_half_width, amplitude),
                        (feature_x, feature_y, 0.5 * amplitude),
                    )
                )
            elif terrain_name == "low_obstacle":
                amplitude = self._interpolate(cfg.low_obstacle_height_range_m, difficulty)
                meshes.append(
                    _box(
                        trimesh,
                        (2.0 * feature_width, 1.2 * feature_width, amplitude),
                        (feature_x, feature_y, 0.5 * amplitude),
                    )
                )

        parameters = {
            "amplitude": amplitude,
            "feature_x": feature_x,
            "feature_y": feature_y,
            "feature_yaw": 0.0,
            "route_yaw": route_yaw,
            "feature_width": feature_width,
            "barrier_half_width": barrier_half_width,
            "traversal_direction": traversal_direction,
        }
        return trimesh.util.concatenate(meshes), parameters


class Go2WTerrainImporter(TerrainImporter):
    """Terrain importer exposing the bank metadata selected by the task."""

    def __init__(self, cfg: TerrainImporterCfg):
        super().__init__(cfg)
        if Go2WTerrainBank.latest_metadata is None:
            raise RuntimeError("Go2W地形库元数据未生成")
        self.bank_metadata = Go2WTerrainBank.latest_metadata


def make_terrain_importer_cfg() -> TerrainImporterCfg:
    """Return the scene configuration for the physical terrain bank."""
    bank_cfg = Go2WTerrainBankCfg()
    bank_cfg.class_type = Go2WTerrainBank
    return TerrainImporterCfg(
        class_type=Go2WTerrainImporter,
        prim_path="/World/terrain",
        terrain_type="generator",
        terrain_generator=bank_cfg,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.30, 0.32, 0.34),
        ),
        debug_vis=False,
    )
