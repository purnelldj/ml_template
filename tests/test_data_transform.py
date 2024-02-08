import numpy as np
import xarray as xr

from cropclassifier.datamodules.snt12_era5_srtm_utils import (
    columns_to_imagelist,
    flatten_varimages,
    imagelist_to_columns,
)


def test_flatten_varimages():
    coords = {"t": [0], "x": [*range(100)], "y": [*range(100)]}
    dims = {"t": 1, "x": 100, "y": 100}
    arrshape = (1, 100, 100)
    np.random.seed(99)
    array1 = np.random.randint(1, 100, size=arrshape)
    array2 = np.random.randint(1, 100, size=arrshape)
    da1 = xr.DataArray(array1, coords=coords, dims=dims, name="var1")
    da2 = xr.DataArray(array2, coords=coords, dims=dims, name="var2")
    dacomb1 = xr.merge([da1, da2])
    array3 = np.random.randint(1, 100, size=arrshape)
    array4 = np.random.randint(1, 100, size=arrshape)
    da3 = xr.DataArray(array3, coords=coords, dims=dims, name="var1")
    da4 = xr.DataArray(array4, coords=coords, dims=dims, name="var2")
    dacomb2 = xr.merge([da3, da4])
    dalist = [dacomb1, dacomb2]
    column = flatten_varimages(dalist, dims, "var1")
    assert column.shape == (2 * 100 * 100, 1)


def test_recreate_image_from_dataarray2():
    coords = {"t": [0], "x": [*range(100)], "y": [*range(100)]}
    dims = {"t": 1, "x": 100, "y": 100}
    arrshape = (1, 100, 100)
    np.random.seed(99)
    array1 = np.random.randint(1, 100, size=arrshape)
    da1 = xr.DataArray(array1, coords=coords, dims=dims, name="var1")
    images = [array1]
    columns = imagelist_to_columns(images, dims)
    dims_out = {"x": 100, "y": 100}
    images_out = columns_to_imagelist(columns, dims_out)
    assert np.array_equal(images_out[0], array1)
    var_out = images_out[0]
    da2 = xr.DataArray(var_out, coords=coords, dims=dims)
    assert da1.equals(da2)


def test_column_to_imagelist():
    images = []
    # the time dimension needs to be fixed for input
    images.append(np.array([[[1, 2], [2, 1], [98, 108], [99, 99], [4, 4], [5, 5]]]))
    images.append(np.array([[[1, 5], [2, 6], [9, 1], [991, 199], [9, 4], [55, 5]]]))
    images.append(np.array([[[1, 6], [6, 1], [8, 0], [9, 9], [9, 9], [5, 5]]]))
    dims = {"x": images[0].shape[1], "y": images[0].shape[2]}
    column = imagelist_to_columns(images, dims)
    assert column.shape == (36, 1)
    images_out = columns_to_imagelist(column, dims)
    for im, im_out in zip(images, images_out):
        assert np.array_equal(im_out, im)


if __name__ == "__main__":
    # test_flatten_varimages()
    # test_column_to_imagelist()
    # test_recreate_image_from_dataarray()
    pass
