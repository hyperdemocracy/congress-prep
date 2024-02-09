from __future__ import annotations
from collections import Counter
import datetime
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element

from bs4 import BeautifulSoup
import pandas as pd
from pydantic import BaseModel


def get_text_or_none(xel: Optional[Element] = None) -> Optional[str]:
    return xel.text if xel is not None else None


class ResAttrs(BaseModel):
    dms_id: str
    key: str
    public_private: str
    resolution_stage: str
    resolution_type: str
    stage_count: Optional[int]
    star_print: str

    @classmethod
    def from_xel(cls, xel: Optional[Element]) -> ResAttrs:
        return ResAttrs(
            dms_id=xel.attrib["dms-id"],
            key=xel.attrib["key"],
            public_private=xel.attrib["public-private"],
            resolution_stage=xel.attrib["resolution-stage"],
            resolution_type=xel.attrib["resolution-type"],
            stage_count=xel.attrib.get("stage-count"),
            star_print=xel.attrib["star-print"],
        )


class ResDublinCore(BaseModel):
    dc_title: str
    dc_publisher: str
    dc_date: Optional[datetime.date]
    dc_format: str
    dc_language: str
    dc_rights: str

    @classmethod
    def from_xel(cls, xel: Optional[Element]) -> ResDublinCore:
        assert xel.tag == "dublinCore"
        ns = {"dc": "http://purl.org/dc/elements/1.1/"}
        return cls(
            dc_title=xel.find("dc:title", namespaces=ns).text,
            dc_publisher=xel.find("dc:publisher", namespaces=ns).text,
            dc_date=xel.find("dc:date", namespaces=ns).text,
            dc_format=xel.find("dc:format", namespaces=ns).text,
            dc_language=xel.find("dc:language", namespaces=ns).text,
            dc_rights=xel.find("dc:rights", namespaces=ns).text,
        )


class ResForm(BaseModel):
    distribution_code: str
    congress: str
    session: str
    legis_num: str
    current_chamber: str
    legis_type: str
    official_title: str

    @classmethod
    def from_xel(cls, xel: Optional[Element]) -> ResForm:
        return cls(
            distribution_code=xel.find("distribution-code").text,
            congress=xel.find("congress").text,
            session=xel.find("session").text,
            legis_num=xel.find("legis-num").text,
            current_chamber=xel.find("current-chamber").text,
            legis_type=xel.find("legis-type").text,
            official_title=xel.find("official-title").text,
        )


class Resolution(BaseModel):
    attrs: ResAttrs
    dublin_core: ResDublinCore
    form: ResForm

    @classmethod
    def from_xel(cls, xel: Optional[Element]) -> Resolution:
        return cls(
            attrs=ResAttrs.from_xel(xel),
            dublin_core = ResDublinCore.from_xel(xel.find("metadata").find("dublinCore")),
            form = ResForm.from_xel(xel.find("form")),
        )


def get_singularized_whitespace(text):
    return " ".join(text.split())


def get_bill_text_v1(xml):
    soup = BeautifulSoup(xml, "xml")
    text = soup.get_text(separator=" ").strip()
    return text


def get_bill_text_v2(xml):
    soup = BeautifulSoup(xml, "xml").find("resolution-body")
    text = soup.get_text(separator=" ").strip()
    return text


def get_bill_text_v3(
    xml,
    tags=[
        "metadata",
        "form",
        "preamble",
        "legis-body",
        "resolution-body",
        "official–title–amendment",
        "impeachment-resolution-signature",
        "attestation",
        "endorsement",
    ],
    singularize_whitespace=True,
):
    soup = BeautifulSoup(xml, "xml")
    texts = []

    for tag in tags:

        element = soup.find(tag)
        if element is None:
            continue

        etext = element.get_text(separator=" ")
        if singularize_whitespace:
            etext = get_singularized_whitespace(etext)

        texts.append(etext)


    text = "\n\n".join(texts)
    return text


def count_tags(xmls: list[str]) -> Counter:
    tags = Counter()
    for xml in xmls:
        root = ET.fromstring(xml)
        tt = tuple([xel.tag for xel in root])
        if '{http://schemas.gpo.gov/xml/uslm}meta' in tt:
            sys.exit(0)
        tags[tt] += 1
    return tags


if __name__ == "__main__":
    congress_hf_path = Path("/Users/galtay/data/congress-hf")
    cn = 117
    xml_file_path = congress_hf_path / f"usc-{cn}-textversions.parquet"
    df_tv_xml = pd.read_parquet(xml_file_path)

    tags = count_tags(df_tv_xml["xml"].tolist())
    sys.exit(0)


    for _, row in df_tv_xml.iterrows():
        xml = row["xml"]
        xml = xml.replace(" & ", " &amp; ")
        text_v1 = get_bill_text_v1(xml)
        text_v2 = get_bill_text_v2(xml)
        text_v3 = get_bill_text_v3(xml)
        tvxml = ET.fromstring(xml)
        res = Resolution.from_xel(tvxml)

        break
        if len(text_v1) > 2048:
            break




