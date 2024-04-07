import datetime
from pathlib import Path
import re

from sqlalchemy import JSON
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column

from congress_prep import utils


class Base(DeclarativeBase):
    pass


class BillStatus(Base):
    __tablename__ = "billstatus"

    legis_id: Mapped[str] = mapped_column(primary_key=True)
    congress_num: Mapped[int]
    legis_type: Mapped[str]
    legis_num: Mapped[int]
    scrape_path: Mapped[str]
    lastmod: Mapped[datetime.datetime]
    bs_xml: Mapped[str]
    bs_json = mapped_column(type_=JSON, nullable=False)


class TextVersions(Base):
    __tablename__ = "textversions"

    tv_id: Mapped[str] = mapped_column(primary_key=True)
    legis_id: Mapped[str]
    congress_num: Mapped[int]
    legis_type: Mapped[str]
    legis_num: Mapped[int]
    legis_version: Mapped[str]
    legis_class: Mapped[str]
    scrape_path: Mapped[str]
    file_name: Mapped[str]
    lastmod: Mapped[datetime.datetime]
    xml_type: Mapped[str]
    tv_xml: Mapped[str]
    tv_txt: Mapped[str]
