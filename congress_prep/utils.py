import re
import pandas as pd


BILLS_PATTERN = re.compile(r"BILLS-(\d{3})([a-zA-Z]+)(\d+)(\w+)\.xml")
PLAW_PATTERN = re.compile(r"PLAW-(\d{3})([a-zA-Z]+)(\d+)\.xml")
FILE_PATTERN = re.compile(
    r"data/(\d{3})/(\w+)/(\w+)/([a-zA-Z]+)(\d+)/fdsys_billstatus\.xml"
)


def metadata_from_unified_row(urow: pd.Series):
    if len(urow["text_versions"]) == 0:
        return {}
    else:
        return {
            "text_id": urow["text_versions"][0]["text_id"],
            "legis_version": urow["text_versions"][0]["legis_version"],
            "legis_class": urow["text_versions"][0]["legis_class"],
            "legis_id": urow["legis_id"],
            "congress_num": urow["congress_num"],
            "legis_type": urow["legis_type"],
            "legis_num": urow["legis_num"],
            "origin_chamber": urow["origin_chamber"],
            "update_date": urow["update_date"],
            "text_date": urow["text_versions"][0]["bs_date"],
            "introduced_date": urow["introduced_date"],
            "sponsor": urow["sponsors"][0]["bioguide_id"],
        }
