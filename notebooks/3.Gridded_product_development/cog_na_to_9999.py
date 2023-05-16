""" Execute the script as follows"""
"""
# Python v3.8.13
# GDAL v3.1.4
# Change input and output path for reading and writing files accordingly

python cog_na_to_9999.py --nodata -9999 --band 'aboveground_biomass' --band 'standard_deviation'
"""

import os
import warnings
from attr import asdict

import numpy
import click
import rasterio
from rasterio.enums import ColorInterp, MaskFlags
from rasterio.io import MemoryFile
from rasterio.rio import options
from rasterio.shutil import copy
from rasterio.vrt import WarpedVRT
import boto3

class NodataParamType(click.ParamType):
    """Nodata type."""

    name = "nodata"

    def convert(self, value, param, ctx):
        """Validate and parse band index."""
        try:
            if value.lower() == "nan":
                return numpy.nan
            elif value.lower() in ["nil", "none", "nada"]:
                return None
            else:
                return float(value)
        except (TypeError, ValueError) as e:
            raise click.ClickException(
                "{} is not a valid nodata value.".format(value)
            ) from e


def has_mask_band(src_dst):
    """Check for mask band in source."""
    if any(
        [
            (MaskFlags.per_dataset in flags and MaskFlags.alpha not in flags)
            for flags in src_dst.mask_flag_enums
        ]
    ):
        return True
    return False

@click.command()
@click.option("--nodata", type=NodataParamType(), metavar="NUMBER|nan", required=True, help="Change Nodata Values.")
@click.option("--band", type=str, multiple=True, help="band names.")
@options.creation_options
def convert(
    nodata,
    band,
    creation_options,
):
    """Create copy."""
    config = {
        "GDAL_NUM_THREADS": "ALL_CPUS",
        "GDAL_TIFF_INTERNAL_MASK": "TRUE",
    }
    
    client = boto3.client('s3')
    my_bucket = 'nasa-maap-data-store'
    prefix = 'file-staging/icesat2-boreal'
    paginator = client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=my_bucket, Prefix=prefix)

    object_list = []
    for page in pages:
        for obj in page['Contents']:
            object_list.append(obj)
    
    for objects in object_list:
        ifile = objects['Key']
        ifileName = ifile.split('/')[-1].split('.')[0]
        
        input = "s3://nasa-maap-data-store/"+ifile
        output = "file-staging/nasa-map/icesat2-boreal/"+ifileName+".tif"
    
        with rasterio.Env(**config):
            with rasterio.open(input) as src_dst:
                meta = src_dst.meta
                meta["nodata"] = nodata
                meta["dtype"] = 'float32'

                output_profile = src_dst.profile

                with MemoryFile() as m:
                    with m.open(**meta) as tmp_dst:
                        if tmp_dst.colorinterp[0] is ColorInterp.palette:
                            try:
                                tmp_dst.write_colormap(1, src_dst.colormap(1))
                            except ValueError:
                                warnings.warn(
                                    "Dataset has `Palette` color interpretation"
                                    " but is missing colormap information"
                                )

                        if has_mask_band(src_dst):
                            tmp_dst.write_mask(src_dst.dataset_mask())

                        arr = src_dst.read()
                        tmp_dst.write(
                            numpy.where(arr != src_dst.nodata, arr, nodata)
                        )
                        del arr

                        tmp_dst._set_all_scales(src_dst.scales)
                        tmp_dst._set_all_offsets(src_dst.offsets)

                        if band and len(band) != len(tmp_dst.indexes):
                            raise ValueError(f"Invalid band option {band} (do not match indexes length)")

                        band = band or [src_dst.descriptions[bidx - 1]  for bidx in tmp_dst.indexes]

                        indexes = tmp_dst.indexes
                        for i, b in enumerate(indexes):
                            if band_name := band[i]:
                                tmp_dst.set_band_description(b, band_name)
                            tmp_dst.update_tags(b, **src_dst.tags(b))

                        tags = src_dst.tags()
                        tags.pop("OVR_RESAMPLING_ALG", None)  # Tags added by rio-cogeo
                        tmp_dst.update_tags(**tags)

                        # We first clear the cache
                        tmp_dst.statistics(1, clear_cache=True)

                        # Should set stats metadata in the tmp_dst
                        for bidx in tmp_dst.indexes:
                            _ = tmp_dst.statistics(bidx)

                        output_profile.update(
                            dict(BIGTIFF=os.environ.get("BIGTIFF", "IF_SAFER"))
                        )
                        if creation_options:
                            output_profile.update(creation_options)

                        keys = [
                            "dtype",
                            "nodata",
                            "width",
                            "height",
                            "count",
                            "crs",
                            "transform",
                        ]
                        for key in keys:
                            output_profile.pop(key, None)

                        output_profile["driver"] = "COG"
                        output_profile["blocksize"] = min(int(output_profile["blockysize"]), int(output_profile["blockysize"]))
                        output_profile.pop("blockxsize", None)
                        output_profile.pop("blockysize", None)
                        output_profile.pop("tiled", None)
                        output_profile.pop("interleave", None)
                        output_profile.pop("photometric", None)

#                         copy(tmp_dst, output, copy_src_overviews=True, **output_profile)

                    client.put_object(Key=output, Bucket=my_bucket, Body=m)


if __name__ == '__main__':
    convert()