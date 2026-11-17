"""局部平面近似下的几何计算（场景尺度 < 100 km）。"""
import numpy as np

M_PER_DEG_LAT = 110574.0
M_PER_DEG_LON_EQ = 111320.0


def enu(lon, lat, lon0, lat0):
    x = (np.asarray(lon) - lon0) * M_PER_DEG_LON_EQ * np.cos(np.radians(lat0))
    y = (np.asarray(lat) - lat0) * M_PER_DEG_LAT
    return x, y


def bearing_deg(lon1, lat1, lon2, lat2):
    """点1 指向点2 的方位，正北 0°，顺时针，[0,360)。"""
    lat_m = (np.asarray(lat1) + np.asarray(lat2)) / 2.0
    dx = (np.asarray(lon2) - np.asarray(lon1)) * M_PER_DEG_LON_EQ * np.cos(np.radians(lat_m))
    dy = (np.asarray(lat2) - np.asarray(lat1)) * M_PER_DEG_LAT
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def dist_m(lon1, lat1, lon2, lat2):
    lat_m = (np.asarray(lat1) + np.asarray(lat2)) / 2.0
    dx = (np.asarray(lon2) - np.asarray(lon1)) * M_PER_DEG_LON_EQ * np.cos(np.radians(lat_m))
    dy = (np.asarray(lat2) - np.asarray(lat1)) * M_PER_DEG_LAT
    return np.hypot(dx, dy)


def angdiff(a, b):
    """有符号角差 a-b，范围 (-180,180]。"""
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def circmean_deg(a, w=None):
    r = np.radians(np.asarray(a))
    s = np.average(np.sin(r), weights=w)
    c = np.average(np.cos(r), weights=w)
    return np.degrees(np.arctan2(s, c)) % 360.0
