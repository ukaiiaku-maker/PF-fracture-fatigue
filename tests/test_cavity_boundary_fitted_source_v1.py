import numpy as np

import arrhenius_fracture.cavity_boundary_fitted_source_v1 as recovery


def test_blocked_nearest_polygon_coordinates_match_direct_operator(monkeypatch):
    polygon=np.asarray(((0.,0.),(1.,0.),(1.,1.),(0.,1.)))
    vectors=np.roll(polygon,-1,axis=0)-polygon
    lengths=np.linalg.norm(vectors,axis=1)
    points=np.asarray(((-.2,.3),(.4,-.1),(1.2,.8),(.7,1.3),(.25,.25),(.5,.5)))

    relative=points[:,None,:]-polygon[None,:,:]
    fraction=np.clip(
        np.einsum("nki,ki->nk",relative,vectors)/(lengths[None,:]**2),0.,1.,
    )
    projections=polygon[None,:,:]+fraction[:,:,None]*vectors[None,:,:]
    distance2=np.einsum(
        "nki,nki->nk",points[:,None,:]-projections,points[:,None,:]-projections,
    )
    expected_nearest=np.argmin(distance2,axis=1)
    rows=np.arange(len(points))

    monkeypatch.setattr(recovery,"MAXIMUM_POINT_FACET_PAIRS_PER_BLOCK",8)
    nearest,nearest_fraction,nearest_projection=(
        recovery._nearest_polygon_coordinates(points,polygon,vectors,lengths)
    )

    assert np.array_equal(nearest,expected_nearest)
    assert np.array_equal(nearest_fraction,fraction[rows,expected_nearest])
    assert np.array_equal(nearest_projection,projections[rows,expected_nearest])
