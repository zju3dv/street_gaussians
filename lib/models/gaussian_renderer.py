import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from lib.utils.sh_utils import eval_sh
from lib.models.gaussian_model import GaussianModel
from lib.models.street_gaussian_model import StreetGaussianModel
from typing import Union

from lib.utils.camera_utils import Camera
from lib.config import cfg

class GaussianRenderer():
    def __init__(
        self,         
    ):
        self.cfg = cfg.render

    def render(
        self, 
        viewpoint_camera: Camera,
        pc: Union[GaussianModel, StreetGaussianModel],
        convert_SHs_python = None, 
        compute_cov3D_python = None, 
        scaling_modifier = None, 
        override_color = None
    ):

        bg_color = [1, 1, 1] if cfg.data.white_background else [0, 0, 0]
        bg_color = torch.tensor(bg_color).float().cuda()
        bg_depth = torch.tensor([0]).float().cuda()
        scaling_modifier = scaling_modifier or self.cfg.scaling_modifier
        convert_SHs_python = convert_SHs_python or self.cfg.convert_SHs_python
        compute_cov3D_python = compute_cov3D_python or self.cfg.compute_cov3D_python
        override_color = override_color or self.cfg.override_color
    
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg_color=bg_color,
            bg_depth=bg_depth,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=self.cfg.debug
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_color, radii, rendered_depth, rendered_acc, rendered_semantic = rasterizer(
            means3D = means3D,
            means2D = means2D,
            opacities = opacity,
            shs = shs,
            colors_precomp = colors_precomp,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp,
            semantics = None,
        )  

        if cfg.mode != 'train':
            rendered_color = torch.clamp(rendered_color, 0., 1.)        
        
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"rgb": rendered_color,
                "acc": rendered_acc,
                "depth": rendered_depth,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii}
