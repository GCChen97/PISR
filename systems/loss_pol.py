import torch



def calc_aop_loss_terms(aops, normals, rays_d_cam, ppa=True, normalized=True, eps=1e-6):
    """_summary_

    Args:
        aops (_type_): Measured AoP
        normals (_type_): Surface normals at camera frame
        rays_d_cam (_type_): Camera rays at camera frame.
        ppa (bool, optional): Whether to consider perspective projection. Defaults to True.
        normalized (bool, optional): Whether to normalize the phase angle loss. Defaults to True.
        eps (float, optional): Small constant for numeric stability. Defaults to 1e-6.
        loss_type (str, optional): Choose a loss function for the phase angle loss.[6d, cos, mvas]
    Returns:
        loss (torch float):
        nrx_sq (torch.tensor): (m^T n)**2
    """
    # PPA
    if ppa:
        vx, vy, vz = [ vi[:,0] for vi in rays_d_cam.split(1,1) ]
    else:
        vx = vy = torch.zeros_like(aops, device=aops.device)
        vz = torch.ones_like(aops, device=aops.device)

    # normal
    nx, ny, nz = [ ni[:,0] for ni in normals.split(1,1) ]

    # diffuse ppa
    sin_phi = torch.sin(aops)
    cos_phi = torch.cos(aops)
    mx = vz * sin_phi
    my = vz * cos_phi
    mz = - (vy*cos_phi + vx*sin_phi)
    mnorm = (mx**2+my**2+mz**2)**0.5 + eps
    mx /= mnorm
    my /= mnorm
    mz /= mnorm

    #specular
    mx_ =             mz * vy - my * vz
    my_ = - mz * vx           + mx * vz
    mz_ = + my * vx - mx * vy
    mnorm_ = (mx_**2+my_**2+mz_**2)**0.5 + eps
    mx_ = mx_ / mnorm_
    my_ = my_ / mnorm_
    mz_ = mz_ / mnorm_

    nrx = mx  * nx + my  * ny + mz  * nz
    nry = mx_ * nx + my_ * ny + mz_ * nz

    nrx_sq = nrx**2
    nry_sq = nry**2

    if normalized:
        norm_square = nrx_sq + nry_sq
        nrx = nrx / (norm_square**0.5 + eps)
        nry = nry / (norm_square**0.5 + eps)

    return nrx, nry

def calc_aop_dop(image, index, y, x):
    I90  = image[index, (y-y%2)  , (x-x%2)  ].view(-1,1)
    I45  = image[index, (y-y%2)  , (x-x%2)+1].view(-1,1)
    I135 = image[index, (y-y%2)+1, (x-x%2)  ].view(-1,1)
    I0   = image[index, (y-y%2)+1, (x-x%2)+1].view(-1,1)

    s0 = (I0 + I90 + I45 + I135)/2
    s1 = I0-I90
    s2 = I45-I135
    
    aop = 0.5*torch.atan2(s2, s1)
    aop[aop < -torch.pi / 2] += torch.pi
    aop[aop >  torch.pi / 2] -= torch.pi

    dop = (s1 ** 2 + s2 ** 2) ** 0.5 / (s0+1e-6)

    return aop, ( torch.abs(s1) <= 1/4095 ) & ( torch.abs(s2) <= 1/4095 ), dop
