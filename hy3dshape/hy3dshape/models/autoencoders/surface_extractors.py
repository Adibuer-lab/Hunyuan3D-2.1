# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

from typing import Union, Tuple, List

import numpy as np
import torch
from skimage import measure

from ...utils import logger


class Latent2MeshOutput:
    def __init__(self, mesh_v=None, mesh_f=None):
        self.mesh_v = mesh_v
        self.mesh_f = mesh_f


class SurfaceExtractor:
    def _compute_box_stat(self, bounds: Union[Tuple[float], List[float], float], octree_resolution: int):
        """
        Compute grid size, bounding box minimum coordinates, and bounding box size based on input 
        bounds and resolution.

        Args:
            bounds (Union[Tuple[float], List[float], float]): Bounding box coordinates or a single 
            float representing half side length.
                If float, bounds are assumed symmetric around zero in all axes.
                Expected format if list/tuple: [xmin, ymin, zmin, xmax, ymax, zmax].
            octree_resolution (int): Resolution of the octree grid.

        Returns:
            grid_size (List[int]): Grid size along each axis (x, y, z), each equal to octree_resolution + 1.
            bbox_min (np.ndarray): Minimum coordinates of the bounding box (xmin, ymin, zmin).
            bbox_size (np.ndarray): Size of the bounding box along each axis (xmax - xmin, etc.).
        """
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

        bbox_min, bbox_max = np.array(bounds[0:3]), np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min
        grid_size = [int(octree_resolution) + 1, int(octree_resolution) + 1, int(octree_resolution) + 1]
        return grid_size, bbox_min, bbox_size

    def run(self, *args, **kwargs):
        """
        Abstract method to extract surface mesh from grid logits.

        This method should be implemented by subclasses.

        Raises:
            NotImplementedError: Always, since this is an abstract method.
        """
        return NotImplementedError

    def __call__(self, grid_logits, **kwargs):
        """
        Process a batch of grid logits to extract surface meshes.

        Args:
            grid_logits (torch.Tensor): Batch of grid logits with shape (batch_size, ...).
            **kwargs: Additional keyword arguments passed to the `run` method.

        Returns:
            List[Optional[Latent2MeshOutput]]: List of mesh outputs for each grid in the batch.
                If extraction fails for a grid, None is appended at that position.
        """
        outputs: List[Latent2MeshOutput] = []
        for i in range(grid_logits.shape[0]):
            try:
                vertices, faces = self.run(grid_logits[i], **kwargs)
            except Exception as exc:
                raise RuntimeError(f"Surface extraction failed for batch index {i}") from exc
            vertices = vertices.astype(np.float32)
            faces = np.ascontiguousarray(faces)
            outputs.append(Latent2MeshOutput(mesh_v=vertices, mesh_f=faces))

        return outputs


class MCSurfaceExtractor(SurfaceExtractor):
    def run(self, grid_logit, *, mc_level, bounds, octree_resolution, **kwargs):
        """
        Extract surface mesh using the Marching Cubes algorithm.

        Args:
            grid_logit (torch.Tensor): 3D grid logits tensor representing the scalar field.
            mc_level (float): The level (iso-value) at which to extract the surface.
            bounds (Union[Tuple[float], List[float], float]): Bounding box coordinates or half side length.
            octree_resolution (int): Resolution of the octree grid.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing:
                - vertices (np.ndarray): Extracted mesh vertices, scaled and translated to bounding 
                  box coordinates.
                - faces (np.ndarray): Extracted mesh faces (triangles).
        """
        vertices, faces, normals, _ = measure.marching_cubes(grid_logit.cpu().numpy(),
                                                             mc_level,
                                                             method="lewiner")
        grid_size, bbox_min, bbox_size = self._compute_box_stat(bounds, octree_resolution)
        vertices = vertices / grid_size * bbox_size + bbox_min
        return vertices, faces


class DMCSurfaceExtractor(SurfaceExtractor):
    def __init__(self):
        super().__init__()
        self.allow_fallback_to_mc = False

    def run(self, grid_logit, *, mc_level, bounds, octree_resolution, method: str = "nagae", **kwargs):
        """
        Extract surface mesh on GPU via isoext's CUDA marching cubes implementation.

        Note: This is not a differentiable path in our inference service; we only require
        a fast, GPU-resident isosurface extraction to avoid the CPU+host-RAM spike caused by
        `skimage.measure.marching_cubes(...)`.

        Args:
            grid_logit (torch.Tensor): 3D grid logits tensor representing the scalar field.
            mc_level (float): The level (iso-value) at which to extract the surface.
            bounds (Union[Tuple[float], List[float], float]): Bounding box coordinates or half side length.
            octree_resolution (int): Resolution of the octree grid.
            method (str): isoext marching cubes method name (default: "nagae").
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing:
                - vertices (np.ndarray): Extracted mesh vertices (bounding box coordinates).
                - faces (np.ndarray): Extracted mesh faces (triangles).
        
        Raises:
            ImportError: If the 'isoext' package is not installed.
        """
        device = grid_logit.device
        if device.type != "cuda":
            if self.allow_fallback_to_mc:
                logger.warning("GPU marching-cubes requires CUDA. Falling back to CPU marching-cubes.")
                return MCSurfaceExtractor().run(
                    grid_logit,
                    mc_level=mc_level,
                    bounds=bounds,
                    octree_resolution=octree_resolution,
                )
            raise RuntimeError("GPU marching-cubes requires CUDA. Set mc_algo to 'mc' or run on a CUDA device.")

        if not hasattr(self, "_isoext"):
            try:
                import isoext

                self._isoext = isoext
            except Exception as exc:
                if self.allow_fallback_to_mc:
                    logger.warning(
                        "GPU marching-cubes unavailable (failed to import isoext). Falling back to CPU marching-cubes."
                    )
                    return MCSurfaceExtractor().run(
                        grid_logit,
                        mc_level=mc_level,
                        bounds=bounds,
                        octree_resolution=octree_resolution,
                    )
                raise ImportError("Please install isoext via `pip install isoext`, or set mc_algo to 'mc'") from exc

        isoext = self._isoext

        grid_size, bbox_min, bbox_size = self._compute_box_stat(bounds, octree_resolution)
        bbox_max = bbox_min + bbox_size

        values = grid_logit
        if values.dtype != torch.float32:
            values = values.to(dtype=torch.float32)
        values = values.contiguous()
        nan_sentinel = float(torch.finfo(torch.float32).max)
        values = torch.nan_to_num(values, nan=nan_sentinel, posinf=nan_sentinel, neginf=-nan_sentinel)
        try:
            grid = isoext.UniformGrid(
                grid_size,
                aabb_min=bbox_min.astype(np.float32).tolist(),
                aabb_max=bbox_max.astype(np.float32).tolist(),
                default_value=nan_sentinel,
            )
            grid.set_values(values)
            verts, faces = isoext.marching_cubes(grid, level=float(mc_level), method=str(method))
        except torch.cuda.OutOfMemoryError:
            if not self.allow_fallback_to_mc:
                raise
            logger.warning("GPU marching-cubes hit CUDA OOM. Falling back to CPU marching-cubes.")
            torch.cuda.empty_cache()
            return MCSurfaceExtractor().run(
                grid_logit,
                mc_level=mc_level,
                bounds=bounds,
                octree_resolution=octree_resolution,
            )
        except RuntimeError as exc:
            if (not self.allow_fallback_to_mc) or ("out of memory" not in str(exc).lower()):
                raise
            logger.warning("GPU marching-cubes hit CUDA OOM. Falling back to CPU marching-cubes.")
            torch.cuda.empty_cache()
            return MCSurfaceExtractor().run(
                grid_logit,
                mc_level=mc_level,
                bounds=bounds,
                octree_resolution=octree_resolution,
            )

        vertices = verts.detach().cpu().numpy()
        faces = faces.detach().cpu().numpy()
        return vertices, faces


SurfaceExtractors = {
    'mc': MCSurfaceExtractor,
    'dmc': DMCSurfaceExtractor,
}
