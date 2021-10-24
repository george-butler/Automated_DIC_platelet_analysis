import numpy as np
import pandas as pd
import math
from collections import defaultdict, Counter
from datetime import datetime
from os.path import join, split, normpath
from skimage.measure import regionprops
from skimage.morphology import binary_dilation
import logging


def highlight_cells(path_in, raw_masks, min_cell_id=None):
    logging.info('================= Highlighting cells =================')
    n_active_features, idx_start_active_features, col_tuple, col_weights, \
        max_displacement, max_absence_interval, out_path \
        = initialize_experiment_parameters(path_in)
    trj = calculate_initial_cell_info(raw_masks, n_active_features, idx_start_active_features,
                                      col_tuple, col_weights, max_displacement, max_absence_interval)
    if min_cell_id is not None:
        reindex_with_min_cell_id(trj, min_cell_id)
    a = trj['particle'].unique()
    for i in a:
        x = trj[trj.particle == i]["wtd_x"].iloc[0]
        y = trj[trj.particle == i]["wtd_y"].iloc[0]
        area_temp = trj[trj.particle == i]["wtd_area"].iloc[0]
        if x < 10:
            trj = trj.drop(trj[trj.particle == i].index)
        if y < 10:
            trj = trj.drop(trj[trj.particle == i].index)
        if x > 960:
            trj = trj.drop(trj[trj.particle == i].index)
        if y > 960:
            trj = trj.drop(trj[trj.particle == i].index)
        if area_temp < 250:
            trj = trj.drop(trj[trj.particle == i].index)
    trj = trj.reset_index(drop=True)
    if min_cell_id is not None:
        reindex_with_min_cell_id(trj, min_cell_id)
    return trj, col_tuple, col_weights


def initialize_experiment_parameters(path_in):
    n_active_features = 8  # Must ALWAYS put the active features in a continuous!!!
    idx_start_active_features = 1  # First active feature
    col_tuple = {'original': ['frame',
                              'y', 'x', 'equivalent_diameter', 'perimeter', 'eccentricity',
                              'orientation_x_2_sin', 'orientation_x_2_cos',
                              'true_solidity',
                              'solidity',
                              'area', 'mean_intensity', 'angle', 'circularity'],
                 'extra': ['bbox_top', 'bbox_left', 'bbox_bottom', 'bbox_right']}
    # Each column will have a "sister" column with prefix 'wtd_', which will represent its weighted version
    col_tuple['weighted'] = ['wtd_{}'.format(x) for x in col_tuple['original']]
    max_displacement = 0
    # Max no. of frames that a cell id may be missing (used mostly in tracking)
    max_absence_interval = 1
    col_weights = defaultdict(lambda: 1,  # Default weight is 1
                              {'frame': max_displacement / max_absence_interval,
                               'equivalent_diameter': 0.75,
                               'perimeter': 0.25,
                               'eccentricity': 5,
                               'orientation_x_2_sin': 5,
                               'orientation_x_2_cos': 5,
                               'true_solidity': 5 * np.pi})

    # Prepare output path
    in_path_part_1, in_path_part_2 = split(normpath(path_in))
    exp_time = datetime.now()  # Experiment time
    out_path = join(in_path_part_1,
                    '{}_Exp_{}-{:02d}-{:02d}T{:02d}{:02d}{:02d}'.format(in_path_part_2,
                                                                        exp_time.year,
                                                                        exp_time.month,
                                                                        exp_time.day,
                                                                        exp_time.hour,
                                                                        exp_time.minute,
                                                                        exp_time.second))
    return n_active_features, idx_start_active_features, col_tuple, col_weights, \
        max_displacement, max_absence_interval, out_path


def calculate_initial_cell_info(raw_masks, n_active_features, idx_start_active_features,
                                col_tuple, col_weights, max_displacement, max_absence_interval):
    features = pd.DataFrame()
    for i_frame in range(raw_masks.shape[2]):
        for region in regionprops(raw_masks[:, :, i_frame], intensity_image=raw_masks[:, :, i_frame]):
            if region.mean_intensity < 1:
                # Skip background (intensity 0)
                continue
            # Append all features
            features = features.append(
                [get_region_info(region, i_frame, col_weights)])
    trj = features.assign(particle=pd.Series(range(1,1+len(features))).values)
    # Reset indexes (current trj has index 0 for all rows; creating new indexes will be useful later)
    trj.reset_index(drop=True, inplace=True)
    return trj


def get_region_info(region, i_frame, col_weights):
    # Compute features
    feat_dict = {'y': region.centroid[0],
                 'x': region.centroid[1],
                 'equivalent_diameter': region.equivalent_diameter,
                 'perimeter': max(1, region.perimeter),
                 'eccentricity': region.eccentricity,
                 'orientation_x_2_sin': np.sin(2 * region.orientation),
                 'orientation_x_2_cos': np.cos(2 * region.orientation),
                 'true_solidity': region.equivalent_diameter / max(1, region.perimeter),
                 'solidity': region.solidity,
                 'area': region.area,
                 'mean_intensity': region.mean_intensity,
                 'angle': region.orientation,
                 'frame': i_frame,
                 'circularity' : (4 * math.pi * region.area) / (region.perimeter * region.perimeter)
                 }
    # Compute weighted features
    weighted_features_list = [('wtd_{}'.format(feat_name), col_weights[feat_name] * feat_val)
                              for feat_name, feat_val in feat_dict.items()]
    feat_dict.update(dict(weighted_features_list))
    # Add extra features, which should not be weighted
    feat_dict.update(dict([('bbox_top', region.bbox[0]),
                           ('bbox_left', region.bbox[1]),
                           ('bbox_bottom', region.bbox[2]),
                           ('bbox_right', region.bbox[3])]))
    return feat_dict

# Get index of cell info in the  trj dataframe, based on a given column value
def get_trj_idx(trj, i_frame, col_name, col_value_list):
    return trj.index[(trj['frame'] == i_frame) & (trj[col_name].isin(col_value_list))]

def reindex_with_min_cell_id(trj, min_cell_id):
    # First, move all cells to higher ids which are safe in the course of id changing
    id_offset = min_cell_id + 1 + max(trj['particle'])
    trj['particle'] = id_offset + trj['particle']
    # Now replace all ids starting from min_cell_id up
    past_cell_ids = set()
    next_cell_id = min_cell_id
    frame_cells = trj.groupby('frame')['particle'].apply(set).to_dict()
    for frame in sorted(list(set(trj['frame'].tolist()))):
        curr_cell_ids = frame_cells[frame]
        for cell_id in sorted(list(curr_cell_ids - past_cell_ids)):
            trj.loc[trj['particle'] == cell_id, 'particle'] = next_cell_id
            next_cell_id += 1
        past_cell_ids |= curr_cell_ids
    trj.reset_index(drop=True, inplace=True)  # Reset indexes, just in case
