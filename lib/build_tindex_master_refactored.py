#!/usr/bin/env python
"""build_tindex_master.py — refactored."""
import os, sys, datetime, argparse, warnings, re
os.environ['USE_PYGEOS'] = '0'

import pandas as pd
import s3fs
import ExtractUtils
from pandarallel import pandarallel

pandarallel.initialize(nb_workers=25, progress_bar=False)
warnings.filterwarnings('ignore')


# =============================================================================
# Path-conversion helpers
# =============================================================================

def local_to_s3(url, user='nathanmthomas', bucket_type='public'):
    if bucket_type == 'public':
        return url.replace('/projects/my-public-bucket', f's3://maap-ops-workspace/shared/{user}')
    return url.replace('/projects/my-private-bucket', f's3://maap-ops-workspace/{user}')


def s3_to_local(url, user='nathanmthomas', bucket_type='private'):
    if bucket_type == 'public':
        return url.replace(f's3://maap-ops-workspace/shared/{user}',
                           f'/projects/my-public-bucket/shared/{user}')
    return url.replace(f's3://maap-ops-workspace/{user}', '/projects/my-private-bucket')


def modification_date(filename):
    return datetime.datetime.fromtimestamp(os.path.getmtime(filename))


# =============================================================================
# TYPE configuration — single source of truth for all type-specific behavior
# =============================================================================

# Each entry maps a TYPE string to behavior:
#   user            : default username when --user not provided
#   ext             : file extension to search for
#   path_template   : path template under bucket; uses {user}, {alg_name},
#                     {dps_id}, {year}, {month}, {day}, {ext} placeholders
#   tile_split_idx  : positional index when splitting filename by '_'.
#                     Use 'last' for ATL08_filt-style (last segment).
#   needs_n_obs     : whether to compute per-file n_obs (ATL08_filt only)

TYPE_CONFIG = {
    'S1': dict(
        user='montesano', ext='_cog.tif',
        path_template='{user}/dps_output/{alg_name}/{dps_id}/{year}/{month}/{day}/**/*.tif',
        tile_split_idx=3,
    ),
    'S1_subtile': dict(
        user='montesano', ext='.tif',
        path_template='{user}/dps_output/{alg_name}/{dps_id}/{year}/{month}/{day}/**/*.tif',
        tile_split_idx='s1_subtile',  # special-cased below
    ),
    'LC': dict(
        user='montesano', ext='_cog.tif',
        path_template='{user}/dps_output/{alg_name}/{dps_id}/{year}/{month}/{day}/**/*.tif',
        tile_split_idx=4,
    ),
    'HLS': dict(
        user='montesano', ext='.tif',
        path_template='{user}/dps_output/{alg_name}/{dps_id}/**/*.tif',
        tile_split_idx=1,
        no_date_in_path=True,
    ),
    'Landsat': dict(
        user='nathanmthomas', ext='_dps.tif',
        path_template='{user}/dps_output/{alg_name}/{dps_id}/{year}/{month}/{day}/**/*_dps.tif',
        tile_split_idx=6,
    ),
    'Topo': dict(
        user='montesano', ext='_stack.tif',
        path_template='{user}/dps_output/{alg_name}/{dps_id}/{year}/{month}/{day}/**/*_stack.tif',
        tile_split_idx=1,
    ),
    'ATL08': dict(
        user='montesano', ext='_30m.csv',
        path_template='{user}/dps_output/{alg_name}/{dps_id}/{year}/{month}/{day}/**/*{seg_str}.csv',
        tile_split_idx=None,        # tile_num='NA'
        focal_field='file',         # dedupe by filename
    ),
    'ATL08_filt': dict(
        user='montesano', ext='.parquet',
        path_template='{user}/dps_output/{alg_name}/{dps_id}/{year}/{month}/{day}/**/*.parquet',
        tile_split_idx='last',
        needs_n_obs=True,
    ),
    'AGB': dict(
        user='montesano', ext='.tif',
        path_template='{user}/dps_output/{alg_name}/{dps_id}/{year}/{month}/{day}/**/*_[0-9][0-9][0-9][0-9][0-9][0-9][0-9].tif',
        tile_split_idx=4,
    ),
    'AGB_UNET': dict(
        user='aliz237',
        ext='.tif',
        path_template='{user}/dps_output/{alg_name}/{dps_id}/{year}/{month}/{day}/**/UNet_*.tif',
        tile_regex=r'UNet_[^_]+_[^_]+_[^_]+_(\d+)_\d+\.tif$',
        content_tag_regex=r'UNet_[^_]+_[^_]+_[^_]+_\d+_(\d{4})\.tif$',   # captures year (e.g., 2024)
    ),
    'AGB_DIFF': dict(
        user='aliz237',
        ext='.tif',
        path_template='{user}/dps_output/{alg_name}/**/AGB_diff_*.tif',   # ← no date subdirs
        tile_regex=r'AGB_diff_\d{4}_\d{4}_(\d+)\.tif$',
        content_tag_regex=r'AGB_diff_(\d{4}_\d{4})_\d+\.tif$',
        no_date_in_path=True,                                              # ← key flag
    ),
    'HT': dict(
        user='montesano', ext='.tif',
        path_template='{user}/dps_output/{alg_name}/{dps_id}/{year}/{month}/{day}/**/*_[0-9][0-9][0-9][0-9][0-9][0-9][0-9].tif',
        tile_split_idx=4,
    ),
    'CACC': dict(
        user='montesano', ext='.tif',
        path_template='{user}/dps_output/{alg_name}/{dps_id}/{year}/{month}/{day}/**/*_[0-9][0-9][0-9][0-9][0-9][0-9][0-9].tif',
        tile_split_idx=3,
    ),
    'TRENDOLS': dict(
        user='montesano', ext='_ols.tif',
        path_template='{user}/dps_output/{alg_name}/{dps_id}/{year}/{month}/{day}/**/*_ols.tif',
        tile_split_idx=3,
    ),
    'TRENDCLASS': dict(
        user='montesano', ext='_kendallclasses.tif',
        path_template='{user}/dps_output/{alg_name}/{dps_id}/{year}/{month}/{day}/**/*_kendallclasses.tif',
        tile_split_idx=3,
    ),
    'TTE': dict(
        user='montesano', ext='.tif',
        path_template='{user}/dps_output/{alg_name}/{dps_id}/{year}/{month}/{day}/**/*.tif',
        tile_split_idx=4,
    ),
    'FORESTAGE': dict(
        user='montesano', ext='_cog.tif',
        path_template='{user}/dps_output/{alg_name}/{dps_id}/{year}/{month}/{day}/**/*.tif',
        tile_split_idx=3,
    ),
    'FORESTAGE100m': dict(
        user='montesano', ext='.tif',
        path_template='{user}/dps_output/{alg_name}/{dps_id}/{year}/{month}/{day}/**/*.tif',
        tile_split_idx=2,
    ),
    'DECIDFRAC': dict(
        user='montesano', ext='_cog.tif',
        path_template='{user}/dps_output/{alg_name}/{dps_id}/{year}/{month}/{day}/**/*.tif',
        tile_split_idx=3,
    ),
    'TRENDNDVI_SLOPE': dict(
        user='pswang', ext='_slope.tif',
        #path_template='{user}/dps_output/{alg_name}/{dps_id}/{year}/{month}/{day}/**/*_slope.tif',
        path_template='{user}/dps_output/{alg_name}/**/*_slope.tif',
        tile_split_idx=0,
    ),
    'TRENDNDVI_PVAL': dict(
        user='pswang', ext='_pval.tif',
        path_template='{user}/dps_output/{alg_name}/**/*_pval.tif',
        tile_split_idx=0,
    ),
}


# =============================================================================
# Search-path construction
# =============================================================================

def build_search_keys(type_name, cfg, args, user):
    """Build list of S3 search patterns for a given TYPE."""
    # Override behaviors that bypass the date-based path entirely
    if args.NO_DPS:
        ext = '.parquet' if 'filt' in type_name or 'ATL08' in type_name else '.tif'
        return [f'{user}/data/{args.dps_identifier}/*{ext}'], ext

    if args.SLIDERULE_OUT and 'filt' in type_name:
        return [f'{user}/data/{args.dps_identifier}/*.parquet'], '.parquet'

    if cfg.get('no_date_in_path'):
        return [cfg['path_template'].format(
            user=user, alg_name=args.alg_name, dps_id=args.dps_identifier
        )], cfg['ext']

    # Standard date-based DPS path
    keys = []
    for year in args.dps_year_list:
        for month in args.dps_month_list:
            for d in range(args.dps_day_min, args.dps_day_max + 1):
                keys.append(cfg['path_template'].format(
                    user=user, alg_name=args.alg_name, dps_id=args.dps_identifier,
                    year=year, month=month, day=f'{d:02d}',
                    seg_str=args.seg_str_atl08,
                ))
    return keys, cfg['ext']


# =============================================================================
# tile_num extraction
# =============================================================================

def extract_tile_num(df, type_name, cfg, ext):
    """Add tile_num and (optionally) file_year based on config."""
    import re
    
    if cfg.get('tile_regex'):
        pattern = cfg['tile_regex']
        df['tile_num'] = df['file'].apply(
            lambda f: (m.group(1) if (m := re.search(pattern, f)) else None)
        )
    
    # Generic content_tag extraction — a per-file identifier beyond tile_num
    # Examples: year ('2024'), year-pair ('2024_2019'), product variant ('L2A'), etc.
    if cfg.get('content_tag_regex'):
        pattern = cfg['content_tag_regex']
        df['content_tag'] = df['file'].apply(
            lambda f: (m.group(1) if (m := re.search(pattern, f)) else None)
        )
    
    if cfg.get('tile_regex'):
        return df
    
    idx = cfg.get('tile_split_idx')
    
    if idx is None:
        df['tile_num'] = 'NA'
        return df
    
    if idx == 'last':
        df['tile_num'] = df['file'].str.split('_').str[-1].str.split(ext, expand=True)[0]
    elif idx == 's1_subtile':
        df['tile_num'] = df['file'].str.split('_tile', expand=True)[1].str.split('-', expand=True)[0]
        df['subtile_num'] = df['file'].str.split('-subtile', expand=True)[1].str.strip('*.tif')
        df['tile_num'] = df['tile_num'].astype(str).astype(int)
    else:
        df['tile_num'] = df['file'].str.split('_', expand=True)[idx].str.strip('*.tif')
    
    return df


def attach_n_obs(df, ext, col_name):
    """For ATL08_filt only — attach per-file row count."""
    print('Multiprocessing n_obs with pandarallel...')
    if ext == '.parquet':
        get_n_obs = lambda f: pd.read_parquet(str(f)).shape[0]
    else:
        get_n_obs = lambda f: pd.read_csv(str(f)).shape[0]

    df['n_obs'] = df[col_name].parallel_apply(get_n_obs)
    return df


# =============================================================================
# Duplicate handling
# =============================================================================

def handle_duplicates(df, focal_fields, return_dups=False, return_creation_time=True):
    if return_creation_time:
        df['creation time'] = df['local_path'].parallel_apply(modification_date)
        df.sort_values(by=['creation time'], ascending=False, inplace=True)

    dropped = df[df.duplicated(subset=focal_fields, keep='first')].copy()
    if len(dropped):
        dropped['status'] = 'dropped'
    else:
        print('\nNo duplicates found.\n')

    df = df.drop_duplicates(subset=focal_fields, keep='first').reset_index(drop=True)

    if 'tile_num' in focal_fields:
        # Only cast to int if all tile_nums are purely numeric;
        # leave as string for geographic IDs like '166W61N'
        if df['tile_num'].astype(str).str.match(r'^\d+$').all():
            df['tile_num'] = df['tile_num'].astype(str).astype(int)
    if 'subtile_num' in focal_fields:
        df['subtile_num'] = df['subtile_num'].apply(
            pd.to_numeric, downcast='signed', errors='coerce')

    print(f"# of duplicate tiles: {len(dropped)}")
    print(f"Final # of tiles: {df.shape[0]}")
    return (df, dropped) if return_dups else df


# =============================================================================
# Per-TYPE processing pipeline
# =============================================================================

def process_type(type_name, args, s3, bucket, has_maap):
    """Run the full tindex pipeline for one TYPE."""
    cfg = TYPE_CONFIG.get(type_name)
    if cfg is None:
        raise ValueError(f"Unknown TYPE: {type_name}. "
                         f"Add an entry to TYPE_CONFIG.")

    user = args.user or cfg['user']

    print(f"\nBuilding tindex for {type_name}")
    print(f"  DPS ID:    {args.dps_identifier}")
    print(f"  User:      {user}")
    print(f"  Years:     {args.dps_year_list}")
    print(f"  Months:    {args.dps_month_list}")
    print(f"  Days:      {args.dps_day_min}-{args.dps_day_max}")
    print(f"  Output:    {args.outdir}")

    # Build the search keys
    if has_maap:
        search_keys, ext = build_search_keys(type_name, cfg, args, user)
    else:
        if args.LOCAL_TEST:
            search_keys = [f"{args.local_dir}/*{args.ends_with_str}"]
            ext = args.ends_with_str
        elif args.root_key:
            search_keys = [f"{args.root_key}/**/*{args.ends_with_str}"]
            ext = args.ends_with_str
        else:
            sys.exit("Need --root_key or --LOCAL_TEST when not on MAAP")

    if args.DEBUG:
        print(f"\nDEBUG:")
        print(f"  search_keys: {search_keys}")
        print(f"  ext:         {ext}")

    # Run S3 glob
    df_parts = [
        pd.DataFrame(s3.glob(os.path.join(bucket, k)), columns=[args.col_name])
        for k in search_keys
    ]
    df = pd.concat(df_parts, ignore_index=True)

    if df.empty:
        print('Nothing found. Check year, month, and search path. Exiting.')
        sys.exit(1)

    # Filter out unwanted strings
    for excl in ['SAMPLE', 'checkpoint']:
        df = df[~df[args.col_name].str.contains(excl)]

    # Build path columns
    df['s3_path']    = ['s3://' + f for f in df[args.col_name]]
    df['local_path'] = [s3_to_local(f, user=user) for f in df[args.col_name]]
    df['file']       = [os.path.basename(f) for f in df[args.col_name]]

    # Tile-num extraction (with one historical override)
    if 'TP' in args.dps_identifier:
        idx = 1 if 'TCC_TP_2020' in args.dps_identifier else 3
        df['tile_num'] = df['file'].str.split('_', expand=True)[idx].str.strip(ext)
    else:
        df = extract_tile_num(df, type_name, cfg, ext)

    if args.content_tag and 'content_tag' in df.columns:
        n_before = len(df)
        df = df[df['content_tag'] == args.content_tag].reset_index(drop=True)
        print(f"content_tag filter: {n_before} → {len(df)} rows (kept only '{args.content_tag}')")
        if df.empty:
            print(f"No files matched content_tag='{args.content_tag}'. Exiting.")
            sys.exit(0)

    # Filter to files matching the requested content year, if specified
    if args.file_year and 'file_year' in df.columns:
        n_before = len(df)
        df = df[df['file_year'] == args.file_year].reset_index(drop=True)
        print(f"file_year filter: {n_before} → {len(df)} rows (kept only '{args.file_year}')")
    
        if df.empty:
            print(f"No files matched file_year='{args.file_year}'. Exiting.")
            sys.exit(0)

    # Attach n_obs for ATL08_filt
    if cfg.get('needs_n_obs'):
        df = attach_n_obs(df, ext, args.col_name)

    # Drop rows with bad tile_num
    n_before = df.shape[0]
    df = df[df['tile_num'].notna()].reset_index(drop=True)
    print(f"Rows: {n_before} → {df.shape[0]} (after NaN removal)")

    if df.empty:
        print('No valid rows remain after tile_num parsing. Exiting.')
        sys.exit(1)

    # Determine focal fields for duplicate handling
    focal_fields = (
        cfg.get('focal_field')
        and [cfg['focal_field']]
        or (['tile_num', 'subtile_num'] if type_name == 'S1_subtile' else ['tile_num'])
    )

    # If a file_year column exists (per-file year, not DPS-processing year), add it to dedup
    if 'file_year' in df.columns:
        focal_fields = focal_fields + ['file_year']

    if 'content_tag' in df.columns:
        focal_fields = focal_fields + ['content_tag']

    # Handle duplicates
    return_creation_time = (type_name != 'S1_subtile')
    try:
        if args.RETURN_DUPS:
            df, dropped = handle_duplicates(df, focal_fields,
                                             return_dups=True,
                                             return_creation_time=return_creation_time)
        else:
            df = handle_duplicates(df, focal_fields,
                                    return_creation_time=return_creation_time)
    except Exception as e:
        # Fallback if creation_time can't be retrieved (cross-user)
        print(f'Falling back without creation time: {e}')
        if args.RETURN_DUPS:
            df, dropped = handle_duplicates(df, focal_fields,
                                             return_dups=True,
                                             return_creation_time=False)
        else:
            df = handle_duplicates(df, focal_fields,
                                    return_creation_time=False)

    # Optionally append to existing tindex
    tag_suffix = f"_{args.content_tag}" if args.content_tag else ""
    out_tindex_fn = os.path.join(args.outdir, f'{type_name}_tindex_master{tag_suffix}.csv')
    if args.tindex_append and os.path.exists(out_tindex_fn):
        print('Appending to existing tindex...')
        df = pd.concat([pd.read_csv(out_tindex_fn), df], ignore_index=True)

    # Write outputs
    if args.RETURN_DUPS and 'dropped' in locals():
        dropped_fn = os.path.splitext(out_tindex_fn)[0] + '_duplicates.csv'
        print(f'Writing duplicates: {dropped_fn}')
        dropped.to_csv(dropped_fn, mode='w')

    print(f'Writing tindex master: {out_tindex_fn}')
    df.to_csv(out_tindex_fn, mode='w', index_label='index')

    # Build mosaic JSON + matches gpkg
    cols_for_mosaic = ['tile_num', 's3_path', 'local_path']
    if type_name == 'S1_subtile':
        cols_for_mosaic = ['subtile_num'] + cols_for_mosaic

    mosaic_json_fn, tindex_matches_gdf = ExtractUtils.build_mosaic_json(
        out_tindex_fn,
        vector_tile_index_path=args.boreal_tile_index_path,
        BAD_TILE_LIST=[],
        cols_list=cols_for_mosaic,
    )

    if args.WRITE_TINDEX_MATCHES_GDF:
        gpkg_fn = os.path.splitext(out_tindex_fn)[0] + '.gpkg'
        print(f'Writing tindex matches gpkg: {gpkg_fn}')
        tindex_matches_gdf.to_file(gpkg_fn, mode='w')

    if 'HLS' in type_name and args.WRITE_MSCOMP_DF:
        ms_comp_name = args.dps_identifier.split('/')[-1]
        print(f'Writing MS composite parameters table for {ms_comp_name}...')
        ExtractUtils.write_mscomp_table(
            f'{args.outdir}/{type_name}_tindex_master.csv',
            RETURN_DF=False,
            MS_COMP_NAME=ms_comp_name,
            mscomp_input_glob_str='output*context.json',
            mscomp_num_scenes_glob_str='master*.json',
            cols_list=['in_tile_num', 'max_cloud', 'start_month_day',
                       'end_month_day', 'start_year', 'end_year'],
        )

    return df


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    valid_types = list(TYPE_CONFIG.keys()) + ['all']
    p = argparse.ArgumentParser()
    p.add_argument('-t', '--type', type=str, choices=valid_types, required=True)
    p.add_argument('-y', '--dps_year', type=str, default='2022')
    p.add_argument('-y_list', '--dps_year_list', nargs='+', type=str, default=None)
    p.add_argument('-m', '--dps_month', type=str, default=None)
    p.add_argument('-m_list', '--dps_month_list', nargs='+', type=str, default=None)
    p.add_argument('-d_min', '--dps_day_min', type=int, default=1)
    p.add_argument('-d_max', '--dps_day_max', type=int, default=31)
    p.add_argument('-alg_name', type=str, required=True)
    p.add_argument('--dps_identifier', type=str, default='master')
    p.add_argument('--user', type=str, default=None)
    p.add_argument('-o', '--outdir', type=str,
                   default='/projects/my-public-bucket/DPS_tile_lists')
    p.add_argument('--seg_str_atl08', type=str, default='_30m')
    p.add_argument('-s', '--ends_with_str', type=str, default='.tif')
    p.add_argument('-b', '--bucket_name', type=str, default=None)
    p.add_argument('-r', '--root_key', type=str, default=None)
    p.add_argument('--col_name', type=str, default='s3_path')
    p.add_argument('-boreal_tile_index_path', type=str,
                   default='/projects/shared-buckets/montesano/databank/boreal_tiles_v004.gpkg')
    p.add_argument('-local_dir', type=str, default=None)
    p.add_argument('--DEBUG', action='store_true')
    p.add_argument('--LOCAL_TEST', action='store_true')
    p.add_argument('--RETURN_DUPS', action='store_true')
    p.add_argument('--SLIDERULE_OUT', action='store_true')
    p.add_argument('--NO_DPS', action='store_true')
    p.add_argument('--tindex_append', action='store_true')
    p.add_argument('--WRITE_TINDEX_MATCHES_GDF', action='store_true')
    p.add_argument('--WRITE_MSCOMP_DF', action='store_true')
    p.add_argument('--file_year', type=str, default=None,
               help='Filter tindex to files matching this year in filename (e.g., 2024)')
    p.add_argument('--content_tag', type=str, default=None,
               help='Filter tindex to files whose extracted content_tag matches this value '
                    '(e.g., "2024" for single-year, "2024_2019" for diff maps)')

    args = p.parse_args()
    args.dps_month_list = args.dps_month_list or [args.dps_month]
    args.dps_year_list  = args.dps_year_list  or [args.dps_year]

    if not args.outdir.startswith('s3://'):
        os.makedirs(args.outdir, exist_ok=True)

    return args


def main():
    args = parse_args()
    s3 = s3fs.S3FileSystem(anon=True)

    try:
        from maap.maap import MAAP
        MAAP()
        has_maap = True
        print('NASA MAAP')
    except ImportError:
        has_maap = False
        print('NASA MAAP unavailable')

    bucket = ('s3://maap-ops-workspace' if has_maap
              else 's3://' + (args.bucket_name or sys.exit("Need --bucket_name off MAAP")))

    types = list(TYPE_CONFIG.keys()) if args.type == 'all' else [args.type]
    for t in types:
        process_type(t, args, s3, bucket, has_maap)


if __name__ == '__main__':
    main()