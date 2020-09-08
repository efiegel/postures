import os
import numpy as np
import pandas as pd
import sklearn
from scipy.spatial import ConvexHull
from itertools import combinations
from progress.bar import Bar

from sklearn.base import BaseEstimator, TransformerMixin
class GeometricFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        d = []

        for _, row in X.iterrows():
            df_x = row.iloc[np.arange(3, 33, 3).tolist()]
            df_y = row.iloc[np.arange(4, 34, 3).tolist()]
            df_z = row.iloc[np.arange(5, 35, 3).tolist()]
            x = np.array(df_x[df_x.notnull()].tolist())
            y = np.array(df_y[df_y.notnull()].tolist())
            z = np.array(df_z[df_z.notnull()].tolist())

            n = len(x)
            digits = 4 # number of decimal places in calculations

            pts = np.vstack((x, y, z)).T

            # length of all lines to the origin
            lengths = []
            for pt in pts:
                length = np.linalg.norm(pt)
                lengths.append(length)
            lengths = np.array(lengths)
            
            # angles between all points and origin
            # e.g. points a, b, c would give 3 angles: a-o-b, a-o-c, b-o-c
            angles = []
            areas = []
            for ang in combinations(pts, 2):
                # Calculate all 3 angles of triangle using 3 sides
                O0 = np.array(ang[0])
                O1 = np.array(ang[1])

                cosine_angle = np.dot(O0, -1 * O1) / (np.linalg.norm(O0) * np.linalg.norm(-1 * O1))
                angle = np.arccos(cosine_angle)
                
                if not np.isnan(angle):
                    angles.append(angle)

                    # Area
                    area = 0.5 * np.linalg.norm(np.cross(O1, -1 * O0))
                    areas.append(area)

            # Determine conex hull of posture, including origin as a data point
            pts = np.vstack((pts, np.array([0, 0, 0])))
            hull = ConvexHull(pts)
            
            # Centroid of convex hull
            cx = np.mean(hull.points[hull.vertices, 0])
            cy = np.mean(hull.points[hull.vertices, 1])
            cz = np.mean(hull.points[hull.vertices, 2])

            d.append({
                'id': int(row.iloc[0]),
                'class': int(row.iloc[1]),
                'user': int(row.iloc[2]),
                'n_markers': n,
                'x_mean': np.mean(x).round(digits),
                'x_std': np.std(x).round(digits),
                'x_min': np.min(x).round(digits),
                'x_max': np.max(x).round(digits),
                'y_mean': np.mean(y).round(digits),
                'y_std': np.std(y).round(digits),
                'y_min': np.min(y).round(digits),
                'y_max': np.max(y).round(digits),
                'z_mean': np.mean(z).round(digits),
                'z_std': np.std(z).round(digits),
                'z_min': np.min(z).round(digits),
                'z_max': np.max(z).round(digits),
                'l_mean': np.mean(lengths).round(digits),
                'l_std': np.std(lengths).round(digits),
                'l_min': np.min(lengths).round(digits),
                'l_max': np.max(lengths).round(digits),
                'ang_mean': np.mean(angles).round(digits),
                'ang_std': np.std(angles).round(digits),
                'ang_min': np.min(angles).round(digits),
                'ang_max': np.max(angles).round(digits),
                'area_mean': np.mean(areas).round(digits),
                'area_std': np.std(areas).round(digits),
                'area_min': np.min(areas).round(digits),
                'area_max': np.max(areas).round(digits),
                'conv_hull_vol': np.round(hull.volume, digits),
                'conv_hull_cx': np.round(cx, digits),
                'conv_hull_cy': np.round(cy, digits),
                'conv_hull_cz': np.round(cz, digits)})
            
        df = pd.DataFrame(d)
        X = df[['id',
                'class',
                'user',
                'n_markers',
                'x_mean',
                'x_std',
                'x_min',
                'x_max',
                'y_mean',
                'y_std',
                'y_min',
                'y_max',
                'z_mean',
                'z_std',
                'z_min',
                'z_max',
                'l_mean',
                'l_std',
                'l_min',
                'l_max',
                'ang_mean',
                'ang_std',
                'ang_min',
                'ang_max',
                'area_mean',
                'area_std',
                'area_min',
                'area_max',
                'conv_hull_vol',
                'conv_hull_cx',
                'conv_hull_cy',
                'conv_hull_cz']]

        return X