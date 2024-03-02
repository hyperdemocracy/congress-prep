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


class BillStatusXml(Base):
    __tablename__ = "billstatus_xml"

    legis_id: Mapped[str] = mapped_column(primary_key=True)
    congress_num: Mapped[int]
    legis_type: Mapped[str]
    legis_num: Mapped[int]
    scrape_path: Mapped[str]
    lastmod: Mapped[datetime.datetime]
    billstatus_xml: Mapped[str]


class BillStatusParsed(Base):
    __tablename__ = "billstatus_parsed"

    legis_id: Mapped[str] = mapped_column(primary_key=True)
    congress_num: Mapped[int]
    legis_type: Mapped[str]
    legis_num: Mapped[int]
    scrape_path: Mapped[str]
    lastmod: Mapped[datetime.datetime]
    billstatus_json = mapped_column(type_=JSON, nullable=False)


class TextVersionsDtdXml(Base):
    __tablename__ = "textversions_dtd_xml"

    text_id: Mapped[str] = mapped_column(primary_key=True)
    legis_id: Mapped[str]
    congress_num: Mapped[int]
    legis_type: Mapped[str]
    legis_num: Mapped[int]
    legis_version: Mapped[str] = mapped_column(nullable=True) # null for plaws
    legis_class: Mapped[str]
    scrape_path: Mapped[str]
    file_name: Mapped[str]
    lastmod: Mapped[datetime.datetime]
    dtd_xml: Mapped[str]


class TextVersionsUslmXml(Base):
    __tablename__ = "textversions_uslm_xml"

    text_id: Mapped[str] = mapped_column(primary_key=True)
    legis_id: Mapped[str]
    congress_num: Mapped[int]
    legis_type: Mapped[str]
    legis_num: Mapped[int]
    legis_version: Mapped[str] = mapped_column(nullable=True) # null for plaws
    legis_class: Mapped[str]
    scrape_path: Mapped[str]
    file_name: Mapped[str]
    lastmod: Mapped[datetime.datetime]
    uslm_xml: Mapped[str]


