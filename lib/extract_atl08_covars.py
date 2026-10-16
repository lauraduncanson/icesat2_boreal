import os
import sys
import re
import glob

import pandas as pd
import geopandas as gpd
import rasterio as rio
import s3fs

import argparse

from maap.maap import MAAP
maap = MAAP(maap_host='api.maap-project.org')

import FilterUtils
import ExtractUtils
import CovariateUtils

# Sentinel-1 band names the models expect (as in the 2019/2020 build_stack_S1 stacks)
S1_SEASONS = ('frozen', 'summer', 'shoulder')
S1_POLS = ('vv', 'vh')
S1_CANONICAL = [f'{p}_median_{s}' for s in S1_SEASONS for p in S1_POLS] + [f'n_{s}' for s in S1_SEASONS]

def parse_rename_map(rename_str):
    '''Parse "old1:new1,old2:new2" into a dict. Written without spaces or quotes so it passes through the DPS shell unchanged.'''
    if rename_str is None or rename_str.strip() in ('', 'None', 'none'):
        return {}
    pairs = [p.split(':', 1) for p in rename_str.split(',') if ':' in p]
    return {old.strip(): new.strip() for old, new in pairs}

def canonical_s1_name(name):
    '''Map a Sentinel-1 band description to the canonical name (e.g. "VV_frozen_median" -> "vv_median_frozen").
    Returns None if the name can't be resolved unambiguously (one polarization and one season, or a count band).'''
    if name is None:
        return None
    tokens = [t for t in re.split(r'[^a-z0-9]+', name.lower()) if t]
    seasons = [s for s in S1_SEASONS if s in tokens]
    if len(seasons) != 1:
        return None
    season = seasons[0]
    if any(t in ('n', 'count', 'num', 'nobs', 'obs') for t in tokens):
        return f'n_{season}'
    pols = [p for p in S1_POLS if p in tokens]
    if len(pols) != 1:
        return None
    stats = [t for t in tokens if t in ('median', 'mean', 'p50', 'avg', 'std', 'min', 'max')]
    stat = 'median' if stats in ([], ['median'], ['p50']) else stats[0]
    return f'{pols[0]}_{stat}_{season}'

def is_s1_stack(path, descriptions):
    return bool(re.search(r'(?i)(^|[/_])(S1|SAR)[_/]', path)) or any(
        d is not None and re.search(r'(?i)(^|[^a-z])v[vh]([^a-z]|$)', d) for d in descriptions)

def resolve_bandnames(covar_fn, rename_map=None, ANON=True):
    '''Band names to use as columns for one covariate stack, plus a table of how each was resolved.
     + explicit rename_map entries win (keyed by the raster's band description, or by "b<N>" for band N)
     + Sentinel-1 stacks are mapped to S1_CANONICAL names when the description is recognizable
     + otherwise the band description is used unchanged
    Raises ValueError if a band has no usable name or two bands resolve to the same name.'''
    rename_map = rename_map or {}
    env = rio.env.Env(AWS_NO_SIGN_REQUEST='YES') if ANON else rio.env.Env()
    with env, rio.open(covar_fn) as src:
        descriptions = list(src.descriptions)
    s1 = is_s1_stack(covar_fn, descriptions)
    rows = []
    for i, d in enumerate(descriptions, 1):
        if d in rename_map:
            name, how = rename_map[d], 'rename_map'
        elif f'b{i}' in rename_map:
            name, how = rename_map[f'b{i}'], 'rename_map (band index)'
        elif s1 and canonical_s1_name(d) is not None:
            name = canonical_s1_name(d)
            how = 'unchanged' if name == d else 'S1 canonical'
        elif d:
            name, how = d, 'unchanged'
        else:
            name, how = None, 'NO NAME'
        rows.append({'band': i, 'description': d, 'column': name, 'how': how})
    table = pd.DataFrame(rows)
    if table.column.isna().any():
        raise ValueError(f'Bands without a name in {covar_fn}: {table[table.column.isna()].band.tolist()}. '
                         f'Give them names with the rename map, e.g. "b1:vv_median_summer".')
    dups = table.column[table.column.duplicated()].unique().tolist()
    if dups:
        raise ValueError(f'Bands resolve to duplicate column names {dups} in {covar_fn}:\n{table}')
    return table.column.tolist(), table

def out_fn_for(s3_atl08_gdf_fn, tile_num):
    '''Output name: input name with "_covars" before the tile number, which is always 6-digit 0-padded.'''
    for pad in (6, 5):
        if f'_{tile_num:0{pad}}.' in s3_atl08_gdf_fn:
            return s3_atl08_gdf_fn.split(f'_{tile_num:0{pad}}.')[0] + f'_covars_{tile_num:06}.parquet'
    raise ValueError(f'Tile number {tile_num} not found at the end of {s3_atl08_gdf_fn}')

def extract_atl08_covars(s3_atl08_gdf_fn: str, tindex_fn_list: list, outdir: str, RETURN_DF=False, EXIT_IF_EMPTY=False, rename_map=None):

    '''Extract values from tiled raster covariates to input tiled sets of atl08 observations using the path of the tiled atl08 geodataframe
     + designed to be multiprocessed with a list of tile nums
     + only works when tile nums of ATL08 matches tile nums of raster covariates
     + to work with any polygon, needs to be updated to read in mosaic of all tiled rasters
     + column names come from the band descriptions; Sentinel-1 names are standardized (see resolve_bandnames)
    '''

    s3 = s3fs.S3FileSystem(anon=True)

    print(s3_atl08_gdf_fn)

    if s3_atl08_gdf_fn.endswith('parquet'):
        atl08 = gpd.read_parquet(s3_atl08_gdf_fn)
    else:
        atl08 = gpd.read_file(s3_atl08_gdf_fn)

    print(atl08.shape)

    # Get file name
    tile_num = int(s3_atl08_gdf_fn.split('_')[-1].split('.')[0]) # tile num is last position in '_' delimited filename
    print(tile_num)

    # This handles 5-zero and 6-0 padded tile_num strings and writes to 6-0 padded strings at end of filename
    out_atl08_covars_fn = out_fn_for(s3_atl08_gdf_fn, tile_num)

    if outdir is not None:
        print(f"Output dir: {outdir}")
        out_atl08_covars_fn = os.path.join(outdir, os.path.basename(out_atl08_covars_fn))

    if EXIT_IF_EMPTY and atl08.shape[0] == 0: sys.exit(f'Tile {tile_num:05} has 0 observations. Exiting.')

    ###################
    # Extract covariates
    ###################
    print(f'\nExtracting values from {len(tindex_fn_list)} sets of raster stacks and appending as columns to atl08 geodataframe...')
    for tindex_fn in tindex_fn_list:
        print(tindex_fn)
        try:
            covar_fn = CovariateUtils.get_stack_fn(tindex_fn, tile_num, user=None, col_name='local_path')
        except IndexError:
            covar_fn = None
        if covar_fn is None:
            sys.exit(f'Tile {tile_num} is not in {tindex_fn}. Exiting.')
        bandnames, band_table = resolve_bandnames(covar_fn, rename_map)
        print(band_table.to_string(index=False))
        overlap = [b for b in bandnames if b in atl08.columns]
        if overlap:
            print(f'\tWARNING: overwriting existing columns {overlap}')
        # This function probably breaks multiprocessing..prob b/c of s3 session?
        atl08 = ExtractUtils.extract_value_gdf_s3(covar_fn, atl08, bandnames, reproject=True)

    atl08.to_parquet(out_atl08_covars_fn)
    print(f'File written:\t{out_atl08_covars_fn}')

    if RETURN_DF:
        return atl08

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-s3_atl08_gdf_fn", type=str, help="The s3 path to the ATL08 geodataframe used for extraction.")
    parser.add_argument("-tindex_fn_list",  nargs='+', type=str, default=None, help="A list of s3 paths to tindex csv files with s3 paths to raster tiles.")
    parser.add_argument("-outdir",  type=str, default=None, help="Output dir")
    parser.add_argument("-band_rename", type=str, default='', help='Optional band renames as "old1:new1,old2:new2" (old = band description, or b<N> for band N). No spaces.')
    parser.add_argument('--RETURN_DF', dest='RETURN_DF', action='store_true', help='Boolean to return a data frame')
    parser.add_argument('--EXIT_IF_EMPTY', dest='EXIT_IF_EMPTY', action='store_true', help='Boolean to exit if an empty data frame; default is to return df with all columns of the covars.')
    parser.set_defaults(RETURN_DF=False)
    parser.set_defaults(EXIT_IF_EMPTY=False)

    args = parser.parse_args()

    extract_atl08_covars(args.s3_atl08_gdf_fn, args.tindex_fn_list, args.outdir, args.RETURN_DF, args.EXIT_IF_EMPTY,
                         rename_map=parse_rename_map(args.band_rename))

if __name__ == "__main__":
    main()
