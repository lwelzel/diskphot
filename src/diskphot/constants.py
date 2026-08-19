"""Constants for the HD 107146 debris-disk transit analysis.

Everything here is either an instrument property, campaign metadata, or an
adopted astrophysical quantity.  Values that plausibly need revisiting are
documented with their provenance so the paper and the code stay in sync.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Campaign metadata (HST/STIS 50CCD coronagraphy, GO 12228 / 14714 / 15221 /
# 15501).  Epoch ordering is chronological and matches the EP{0..7} FITS files.
# --------------------------------------------------------------------------- #
N_EPOCHS: int = 8

EPOCH_DATES: tuple[str, ...] = (
    "2011-02-22",
    "2011-05-03",
    "2017-03-26",
    "2017-07-28",
    "2018-03-16",
    "2018-07-29",
    "2018-12-20",
    "2019-05-28",
)

#: Total on-target integration time per epoch [s].  EP0/EP1 from GO 12228;
#: EP2..EP7 are the follow-on epochs, ~8314 s each (4 orbits x 5 x 479 s
#: exposures, after masked-median combination; see the observations section).
EPOCH_EXPTIMES_S: tuple[float, ...] = (
    1588.0,
    1662.8,
    8314.0,
    8314.0,
    8314.0,
    8314.0,
    8314.0,
    8314.0,
)

#: Background RMS of the reduced count-rate images [counts s^-1 px^-1],
#: measured far from star and galaxy (paper Fig. GS02 caption).  Used as a
#: fallback when the per-epoch background is not measured from the data.
SIGMA_BG_DEFAULT: float = 3.0e-3

#: Detector gain [e- / count] used in the photon-noise term of the per-pixel
#: variance.  The reduced images are count-rate images; with gain g and
#: exposure time t the Poisson variance of a rate r is r / (g t).  The
#: analysis in the manuscript treats counts as electrons (g = 1); STIS
#: coronagraphy is usually taken with CCDGAIN = 4, so this is configurable.
GAIN_DEFAULT: float = 1.0

#: STIS CCD plate scale [arcsec / px].  The instrument handbook value is
#: 0.05072"/px; note the manuscript currently quotes 50.77 mas (Sec. 2) and
#: 50.077 mas (Fig. GS02 caption) -- reconcile there, configure here.
PIXEL_SCALE_ARCSEC: float = 0.05072

# --------------------------------------------------------------------------- #
# HD 107146 system (Gaia DR3; adopted for all pixel -> au conversions).
# --------------------------------------------------------------------------- #
GAIA_DR3_PARALLAX_MAS: float = 36.404
GAIA_DR3_PARALLAX_ERR_MAS: float = 0.023
DISTANCE_PC: float = 1000.0 / GAIA_DR3_PARALLAX_MAS  # ~27.47 pc
#: Proper motion (pmRA*, pmDec) [mas / yr], Gaia DR3.
PROPER_MOTION_MASYR: tuple[float, float] = (-174.7, -149.0)

# --------------------------------------------------------------------------- #
# Disk scattered-light model (Schneider best-fit model FITS).
# --------------------------------------------------------------------------- #
#: Shift [px] that co-aligns the model image with the NUP-reduced epochs,
#: concentric with the (occulted) star ("translate the model image by
#: (-540.45, -541.08) pixels", G. Schneider, priv. comm.).
DISK_MODEL_SHIFT_XY: tuple[float, float] = (-540.45, -541.08)

#: The scattered-light model is not defined interior to this stellocentric
#: radius [px]; the ring surface-brightness peak is searched outside it.
DISK_MODEL_INNER_VALID_RADIUS_PX: float = 40.0

# --------------------------------------------------------------------------- #
# Grain properties adopted in the discussion (column-density conversion).
# --------------------------------------------------------------------------- #
#: Effective mean grain radius for the tau -> Sigma conversion [cm] (3.7 um).
GRAIN_RADIUS_MEAN_CM: float = 3.7e-4
#: Internal (solid) grain density [g cm^-3].
GRAIN_SOLID_DENSITY: float = 1.5
#: Geometric-optics extinction efficiency (a_min >> lambda / 2 pi).
Q_EXT: float = 2.0

AU_PER_PC_ARCSEC: float = 1.0  # 1 arcsec at 1 pc == 1 au (definition)


def arcsec_to_au(theta_arcsec: np.ndarray | float) -> np.ndarray | float:
    """Convert an angular separation [arcsec] to projected au at HD 107146."""
    return np.asarray(theta_arcsec) * DISTANCE_PC


def pixels_to_arcsec(r_px: np.ndarray | float) -> np.ndarray | float:
    """Convert a separation in detector pixels to arcsec."""
    return np.asarray(r_px) * PIXEL_SCALE_ARCSEC
