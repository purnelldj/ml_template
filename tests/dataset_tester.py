import os

import numpy as np

from cropclassifier.datamodules.snt12_era5_srtm_utils import rioxload


def test_snt12_era5_srtm_utils():
    dsdir = "datasets/snt12_era5_srtm/"
    files = os.listdir(dsdir)
    fullfiles = [dsdir + file for file in files]
    varlist = []
    varlist += ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12", "B8A"]
    varlist += ["temperature-mean", "precipitation-flux"]
    varlist += ["VV", "VH"]
    varlist += ["altitude", "slope"]
    for file, ffile in zip(files, fullfiles):
        d = rioxload(ffile)
        print(file)
        # now go through var by var and timetstep by timestep
        for var in varlist:
            try:
                nparr = d[var].values
            except KeyError:
                print(f"var {var} is missing for file {file}")
            mask = nparr == 65535
            if np.sum(mask) == nparr.shape[0] * nparr.shape[1] * nparr.shape[2]:
                print(f"var {var} is missing for file {file}")
                continue
            for i in range(12):
                mask_red = nparr[i, :, :] == 65535
                if np.sum(mask_red) == nparr.shape[1] * nparr.shape[2]:
                    print(f"var {var} is missing at time index {i} for file {file}")


if __name__ == "__main__":
    test_snt12_era5_srtm_utils()
