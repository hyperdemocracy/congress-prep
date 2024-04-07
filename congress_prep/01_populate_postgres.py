"""
Insert scraped legislation xml files into postgres.
"""

import json
from pathlib import Path
import re
import rich
from typing import Union
from typing import Optional

import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import insert

from congress_prep import orm_mod
from congress_prep import utils
from congress_prep.bill_status_mod import BillStatus
from congress_prep.textversions_mod import get_bill_text_v4


def get_session(conn_str: str, echo=False):
    engine = create_engine(conn_str, echo=echo)
    Session = sessionmaker(engine)
    return Session


def reset_tables(conn_str: str, echo=False):
    engine = create_engine(conn_str, echo=echo)
    orm_mod.Base.metadata.drop_all(engine)
    orm_mod.Base.metadata.create_all(engine)


def create_tables(conn_str: str, echo=False):
    engine = create_engine(conn_str, echo=echo)
    orm_mod.Base.metadata.create_all(engine)


def compile_query(query):
    return str(query.statement.compile(dialect=postgresql.dialect()))


def upsert(
    session: sqlalchemy.orm.Session,
    table: sqlalchemy.Table,
    rows: list[dict],
    lastmod_col: str = "lastmod",
    no_update_cols: Optional[list[str]] = None,
):
    if no_update_cols is None:
        no_update_cols = []
    stmt = insert(table).values(rows)
    update_cols = [
        c.name
        for c in table.c
        if c not in list(table.primary_key.columns) and c.name not in no_update_cols
    ]
    on_conflict_stmt = stmt.on_conflict_do_update(
        index_elements=table.primary_key.columns,
        set_={k: getattr(stmt.excluded, k) for k in update_cols},
    )
    session.execute(on_conflict_stmt)
    session.commit()


def upsert_billstatus(
    congress_scraper_path: Union[str, Path], conn_str: str, batch_size: int = 1000
):
    """Upsert billstatus xml files into postgres

    Args:
        congress_scraper_path: should have "cache" and "data" as subdirectories
        conn_str: postgres connection string
        batch_size: number of billstatus files to upsert at once
    """

    data_path = Path(congress_scraper_path) / "data"
    Session = get_session(conn_str)

    rows = []
    ibatch = 0
    for path_object in data_path.rglob("fdsys_billstatus.xml"):
        path_str = str(path_object.relative_to(congress_scraper_path))
        if match := re.match(utils.FILE_PATTERN, path_str):
            congress_num, legis_class, legis_type, _, legis_num = match.groups()
        else:
            rich.print("billstatus oops: {}".format(path_object))
            continue

        lastmod_path = path_object.parent / "fdsys_billstatus-lastmod.txt"
        lastmod_str = lastmod_path.read_text()
        xml_str = path_object.read_text().strip()
        bs = BillStatus.from_xml_str(xml_str)

        row = {
            "legis_id": "{}-{}-{}".format(congress_num, legis_type, legis_num),
            "congress_num": int(congress_num),
            "legis_type": legis_type,
            "legis_num": int(legis_num),
            "scrape_path": path_str,
            "lastmod": lastmod_str,
            "bs_xml": xml_str,
            "bs_json": json.loads(bs.model_dump_json()),
        }
        rows.append(row)

        if len(rows) >= batch_size:
            rich.print(f"upserting billstatus batch {ibatch} with {len(rows)} rows.")
            with Session() as session:
                upsert(session, orm_mod.BillStatus.__table__, rows)
            rows = []
            ibatch += 1

    if len(rows) > 0:
        rich.print(f"upserting billstatus batch {ibatch} with {len(rows)} rows.")
        with Session() as session:
            upsert(session, orm_mod.BillStatus.__table__, rows)


def upsert_textversions(
    congress_scraper_path: Union[str, Path], conn_str: str, batch_size: int = 1000
):
    """Upsert textversions xml files into postgres

    Args:
        congress_scraper_path: should have "cache" and "data" as subdirectories
        conn_str: postgres connection string
        batch_size: number of billstatus files to upsert at once
    """

    data_path = Path(congress_scraper_path) / "data"
    Session = get_session(conn_str)

    rows = []
    ibatch = 0
    for path_object in data_path.rglob("*.xml"):
        path_str = str(path_object.relative_to(congress_scraper_path))

        if "/uslm/" in path_str:
            xml_type = "uslm"
        else:
            xml_type = "dtd"

        if match := re.match(utils.BILLS_PATTERN, path_object.name):
            legis_class = "bills"
            congress_num, legis_type, legis_num, legis_version = match.groups()

        elif match := re.match(utils.PLAW_PATTERN, path_object.name):
            legis_class = "plaw"
            legis_version = "plaw"
            congress_num, legis_type, legis_num = match.groups()

        else:
            continue

        lastmod_path = path_object.parent / (
            path_object.name.split(".")[0] + "-lastmod.txt"
        )
        lastmod_str = lastmod_path.read_text()
        xml = path_object.read_text().strip()

        row = {
            "tv_id": "{}-{}-{}-{}-{}".format(
                congress_num, legis_type, legis_num, legis_version, xml_type
            ),
            "legis_id": "{}-{}-{}".format(congress_num, legis_type, legis_num),
            "congress_num": int(congress_num),
            "legis_type": legis_type,
            "legis_num": int(legis_num),
            "legis_version": legis_version,
            "legis_class": legis_class,
            "scrape_path": path_str,
            "file_name": Path(path_str).name,
            "lastmod": lastmod_str,
            "xml_type": xml_type,
            "tv_xml": xml,
            "tv_txt": get_bill_text_v4(xml),
        }
        rows.append(row)

        if len(rows) >= batch_size:
            rich.print(f"upserting textversions batch {ibatch} with {len(rows)} rows.")
            with Session() as session:
                upsert(session, orm_mod.TextVersions.__table__, rows)
            rows = []
            ibatch += 1

    if len(rows) > 0:
        rich.print(f"upserting textversions batch {ibatch} with {len(rows)} rows.")
        with Session() as session:
            upsert(session, orm_mod.TextVersions.__table__, rows)


def create_unified(conn_str: str):
    """Join billstatus and textversions data.
    Note that this uses the dtd xml text version not the uslm xml versions.

    BS = billstatus
    TV = textversions

    billstatus xml files have an array of textversion info (not the actual text).
    each textversions xml file has one version of the text for a bill.

    Args:
        conn_str: postgres connection string
    """

    sql = """
    drop table if exists unified;
    create table unified as (

    with

    -- turn BS textversion array with N entries into N rows
    bs_tvs_v1 as (
      select
        legis_id,
        json_array_elements(bs_json->'text_versions') as bs_tv
      from billstatus
    ),

    -- pull file name from BS textversion for joining to TV
    bs_tvs_v2 as (
      select
        legis_id,
        bs_tv,
        bs_tv->'url' as url,
        split_part(bs_tv->>'url', '/', -1) as file_name
      from bs_tvs_v1
    ),

    -- join BS and TV textversions. keep only dtd xml text versions
    jnd_tvs as (
      select
        textversions.*,
        bs_tv
      from bs_tvs_v2
      join textversions
      on bs_tvs_v2.file_name = textversions.file_name
      where xml_type = 'dtd'
    ),

    -- group TV info by legis_id
    tvs as (
      select
        legis_id,
        json_agg(
          json_build_object(
            'tv_id', tv_id,
            'legis_id', legis_id,
            'congress_num', congress_num,
            'legis_type', legis_type,
            'legis_num', legis_num,
            'legis_version', legis_version,
            'legis_class', legis_class,
            'scrape_path', scrape_path,
            'file_name', file_name,
            'lastmod', lastmod,
            'xml_type', xml_type,
            'tv_xml', tv_xml,
            'tv_txt', tv_txt,
            'bs_tv', bs_tv
          ) order by lastmod desc
        ) as tvs
      from jnd_tvs
      group by legis_id
    )

    -- join billstatus info with text versions
    select billstatus.*, tvs.tvs from billstatus join tvs
    on billstatus.legis_id = tvs.legis_id
    )
    """

    engine = create_engine(conn_str, echo=True)
    with engine.connect() as conn:
        with conn.begin():
            result = conn.execute(text(sql))


if __name__ == "__main__":

    conn_str = "postgresql+psycopg2://galtay@localhost:5432/galtay"
    congress_scraper_path = Path("/Users/galtay/data/congress-scraper")

    #    reset_tables(conn_str)
    #    upsert_billstatus(congress_scraper_path, conn_str)
    #    upsert_textversions(congress_scraper_path, conn_str)
    create_unified(conn_str)

    engine = create_engine(conn_str, echo=True)
    df = pd.read_sql("select * from unified limit 1", con=engine)
