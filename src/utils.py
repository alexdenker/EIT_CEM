import numpy as np
from scipy.interpolate import interpn, NearestNDInterpolator


def image_to_mesh(x, mesh_pos, fill_value=1.0):
    """
    Interpolate image x to mesh given by mesh positions. 

    x: [1, H, W] numpy array 
    
    """
    assert len(x.shape) == 3, f"wrong shape of image: {x.shape}"

    radius = np.max(np.abs(mesh_pos))

    pixcenter_x = pixcenter_y = np.linspace(-radius, radius, x.shape[-1])
    X, Y = np.meshgrid(pixcenter_x, pixcenter_y, indexing="ij")
    sigma = interpn(
        [pixcenter_x, pixcenter_y],
        np.flipud(x[0]).T, 
        mesh_pos,
        bounds_error=False,
        fill_value=fill_value,
        method="nearest",
    )
    
    return sigma


def interpolate_mesh_to_mesh(x, mesh_pos1, mesh_pos2):
    interpolator = NearestNDInterpolator(mesh_pos1, x)

    sigma = interpolator(mesh_pos2[:, 0], mesh_pos2[:, 1])

    return sigma


def transform_data(U, MeasPat):
    """
    This method transform the the measured data by the KIT4 system (differences between neighbouring electrodes)
    to our format (potential at each electrode)
    
    U: Measurements in KIT4 format
    MeasPat: Measurement pattern 

    """

    Bf = np.vstack([MeasPat, np.ones(MeasPat.shape[-1])])

    U_ = []
    for i in range(U.shape[0]):
        U_sol, res, _, _ = np.linalg.lstsq(Bf, np.hstack([U[i, :], np.array([0])]))
        U_.append(U_sol)

    Uel = np.stack(U_)

    return Uel
