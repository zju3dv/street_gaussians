import math
import torch
import time
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
if __name__ == '__main__':
    viewpoint_camera = {
        'image_height': 375,
        'image_width': 1242,
        'FoVx': 1.416,
        'FoVy': 0.506,
        'world_view_transform': torch.tensor([[ 0.9598,  0.0081,  0.2806,  0.0000],
        [-0.0123,  0.9998,  0.0134,  0.0000],
        [-0.2804, -0.0163,  0.9597,  0.0000],
        [-2.0954, -0.0935,  4.9320,  1.0000]]).cuda(),
        'full_proj_transform': torch.tensor([[ 1.1205,  0.0312,  0.2806,  0.2806],
        [-0.0144,  3.8661,  0.0134,  0.0134],
        [-0.3274, -0.0632,  0.9598,  0.9597],
        [-2.4464, -0.3614,  4.9225,  4.9320]]).cuda(),
        'camera_center': torch.tensor([ 6.2808e-01,  1.4572e-03, -5.3226e+00]).cuda(),
    }
    
    bg_color = torch.zeros(3).cuda()
    scaling_modifier = 1.0
    active_sh_degree = 0
    tanfovx = math.tan(viewpoint_camera['FoVx'] * 0.5)
    tanfovy = math.tan(viewpoint_camera['FoVy'] * 0.5)
    
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera['image_height']),
        image_width=int(viewpoint_camera['image_width']),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera['world_view_transform'],
        projmatrix=viewpoint_camera['full_proj_transform'],
        sh_degree=active_sh_degree,
        campos=viewpoint_camera['camera_center'],
        prefiltered=False,
        debug=True
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    num_points = 10000
    means3D = torch.rand(num_points, 3).cuda()
    means2D = torch.rand(num_points, 3).cuda()
    shs = torch.rand(num_points, 4, 3).cuda()
    colors_precomp = None
    opacity = torch.rand(num_points, 1).cuda()
    scales = torch.rand(num_points, 3).cuda()
    rotations = torch.rand(num_points, 4).cuda()
    rotations[:, 0] = 1
    cov3D_precomp = None
    
    torch.cuda.synchronize()    
    s = time.time()
    rendered_image, radii, rendered_depth, rendered_alpha, rendered_semantic = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        semantics=None
    )
    torch.cuda.synchronize()
    e = time.time()

    print(f'Test 1: Pass diff_gaussian_rasterization !, using {1000* (e - s)}ms')
        
    semantics = torch.rand(num_points, 15).cuda()
    
    torch.cuda.synchronize()    
    s = time.time()
    rendered_image, radii, rendered_depth, rendered_alpha, rendered_semantic = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        semantics=semantics
    )
    torch.cuda.synchronize()
    e = time.time()

    print(f'Test 2: Pass diff_gaussian_rasterization !, using {1000* (e - s)}ms')