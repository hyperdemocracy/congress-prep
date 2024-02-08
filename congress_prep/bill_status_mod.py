"""
https://github.com/usgpo/bill-status/tree/main
https://www.congress.gov/help/legislative-glossary
"""
import datetime
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element

from pydantic import BaseModel


def get_text_or_none(element: Optional[Element] = None) -> Optional[str]:
    return element.text if element is not None else None


class Activity(BaseModel):
    name: str
    date: datetime.datetime


def get_activities(element: Optional[Element]) -> list[Activity]:
    result = []
    if element is None:
        return result
    assert element.tag == "activities"
    for item in element:
        if item.tag != "item":
            continue
        activity = Activity(
            name=get_text_or_none(item.find("name")),
            date=get_text_or_none(item.find("date")),
        )
        result.append(activity)
    return result


class Subcommittee(BaseModel):
    system_code: str
    name: str
    activities: list[Activity]


def get_subcommittees(element: Optional[Element]) -> list[Subcommittee]:
    result = []
    if element is None:
        return result
    assert element.tag == "subcommittees"
    for item in element:
        if item.tag != "item":
            continue
        subcommittee = Subcommittee(
            system_code=get_text_or_none(item.find("systemCode")),
            name=get_text_or_none(item.find("name")),
            activities=get_activities(item.find("activities")),
        )
        result.append(subcommittee)
    return result


class Committee(BaseModel):
    system_code: str
    name: str
    chamber: Optional[str] = None
    type: Optional[str] = None
    subcommittees: list[Subcommittee] = []
    activities: list[Activity] = []


def get_committees(element: Optional[Element]) -> list[Committee]:
    result = []
    if element is None:
        return result
    assert element.tag == "committees"
    for item in element:
        if item.tag != "item":
            continue
        committee = Committee(
            system_code=get_text_or_none(item.find("systemCode")),
            name=get_text_or_none(item.find("name")),
            chamber=get_text_or_none(item.find("chamber")),
            type=get_text_or_none(item.find("type")),
            subcommittees=get_subcommittees(item.find("subcommittees")),
            activities=get_activities(item.find("activities")),
        )
        result.append(committee)
    return result


class CommitteeReport(BaseModel):
    citation: str


def get_committee_reports(element: Optional[Element]) -> list[CommitteeReport]:
    result = []
    if element is None:
        return result
    assert element.tag == "committeeReports"
    for item in element:
        if item.tag != "committeeReport":
            continue
        committee_report = CommitteeReport(
            citation=get_text_or_none(item.find("citation")),
        )
        result.append(committee_report)
    return result


class SourceSystem(BaseModel):
    name: str
    code: Optional[str] = None


def get_source_system(element: Optional[Element]) -> Optional[SourceSystem]:
    if element is None:
        return None
    else:
        assert element.tag == "sourceSystem"
        return SourceSystem(
            name=get_text_or_none(element.find("name")),
            code=get_text_or_none(element.find("code")),
        )


class Action(BaseModel):
    action_date: datetime.date
    text: str
    type: Optional[str] = None
    action_code: Optional[str] = None
    source_system: Optional[SourceSystem] = None
    committees: list[Committee] = []


def get_action(element: Optional[Element]) -> Optional[Action]:
    if element is None:
        return None
    action = Action(
        action_date=get_text_or_none(element.find("actionDate")),
        text=get_text_or_none(element.find("text")),
        type=get_text_or_none(element.find("type")),
        action_code=get_text_or_none(element.find("actionCode")),
        source_system=get_source_system(element.find("sourceSystem")),
        committees=get_committees(element.find("committees")),
    )
    return action


def get_actions(element: Optional[Element]) -> list[Action]:
    result = []
    if element is None:
        return result
    assert element.tag == "actions"
    for item in element:
        if item.tag != "item":
            continue
        action = get_action(item)
        if action is not None:
            result.append(action)
    return result


class RelationshipDetail(BaseModel):
    type: str
    identified_by: str


class RelatedBill(BaseModel):
    title: str
    congress: int
    number: int
    type: str
    latest_action: Action
    relationship_details: list[RelationshipDetail]


def get_relationship_details(element: Optional[Element]) -> list[RelationshipDetail]:
    result = []
    if element is None:
        return result
    assert element.tag == "relationshipDetails"
    for item in element:
        if item.tag != "item":
            continue
        relationship_detail = RelationshipDetail(
            type=get_text_or_none(item.find("type")),
            identified_by=get_text_or_none(item.find("identifiedBy")),
        )
        result.append(relationship_detail)
    return result


def get_related_bills(element: Optional[Element]) -> list[RelatedBill]:
    result = []
    if element is None:
        return result
    assert element.tag == "relatedBills"
    for item in element:
        if item.tag != "item":
            continue
        related_bill = RelatedBill(
            title=get_text_or_none(item.find("title")),
            congress=get_text_or_none(item.find("congress")),
            number=get_text_or_none(item.find("number")),
            type=get_text_or_none(item.find("type")),
            latest_action = get_action(item.find("latestAction")),
            relationship_details = get_relationship_details(item.find("relationshipDetails")),
        )
        result.append(related_bill)
    return result


class Title(BaseModel):
    title_type: str
    title: str
    chamber_code: Optional[str] = None
    chamber_name: Optional[str] = None
    bill_text_version_name: Optional[str] = None
    bill_text_version_code: Optional[str] = None


def get_titles(element: Element) -> list[Title]:
    result = []
    if element is None:
        return result
    assert element.tag == "titles"
    for item in element:
        if item.tag != "item":
            continue
        title = Title(
            title_type=get_text_or_none(item.find("titleType")),
            title=get_text_or_none(item.find("title")),
            chamber_code=get_text_or_none(item.find("chamberCode")),
            chamber_name=get_text_or_none(item.find("chamberName")),
            bill_text_version_name=get_text_or_none(item.find("billTextVersionName")),
            bill_text_version_code=get_text_or_none(item.find("billTextVersionCode")),
        )
        result.append(title)
    return result


class Identifiers(BaseModel):
    bioguide_id: str
    lis_id: Optional[str] = None
    gpo_id: Optional[str] = None


def get_identifiers(element: Optional[Element]) -> Optional[Identifiers]:
    if element is None:
        return None
    else:
        assert element.tag == "identifiers"
        return Identifiers(
            lis_id=get_text_or_none(element.find("lisID")),
            bioguide_id=get_text_or_none(element.find("bioguideId")),
            gpo_id=get_text_or_none(element.find("gpoId")),
        )


class Sponsor(BaseModel):
    bioguide_id: str
    full_name: str
    first_name: str
    last_name: str
    party: str
    state: str
    identifiers: Optional[Identifiers] = None
    middle_name: Optional[str] = None
    district: Optional[str] = None
    is_by_request: Optional[str] = None


def get_sponsors(element: Element) -> list[Sponsor]:
    result = []
    if element is None:
        return result
    assert element.tag == "sponsors"
    for item in element:
        if item.tag != "item":
            continue
        # some amendment sponsors have only a name like "Rules Committee"
        # skip these for now
        if [el.tag for el in item] == ["name"]:
            continue
        sponsor = Sponsor(
            bioguide_id=get_text_or_none(item.find("bioguideId")),
            full_name=get_text_or_none(item.find("fullName")),
            first_name=get_text_or_none(item.find("firstName")),
            last_name=get_text_or_none(item.find("lastName")),
            party=get_text_or_none(item.find("party")),
            state=get_text_or_none(item.find("state")),
            middle_name=get_text_or_none(item.find("middleName")),
            district=get_text_or_none(item.find("district")),
            is_by_request=get_text_or_none(item.find("isByRequest")),
            identifiers=get_identifiers(item.find("identifiers")),
        )
        result.append(sponsor)
    return result


class Cosponsor(BaseModel):
    bioguide_id: str
    full_name: str
    first_name: str
    last_name: str
    party: str
    state: str
    middle_name: Optional[str] = None
    district: Optional[str] = None
    sponsorship_date: datetime.date
    is_original_cosponsor: bool


def get_cosponsors(element: Element) -> list[Cosponsor]:
    result = []
    if element is None:
        return result
    assert element.tag == "cosponsors"
    for item in element:
        if item.tag != "item":
            continue
        cosponsor = Cosponsor(
            bioguide_id=get_text_or_none(item.find("bioguideId")),
            full_name=get_text_or_none(item.find("fullName")),
            first_name=get_text_or_none(item.find("firstName")),
            last_name=get_text_or_none(item.find("lastName")),
            party=get_text_or_none(item.find("party")),
            state=get_text_or_none(item.find("state")),
            middle_name=get_text_or_none(item.find("middleName")),
            district=get_text_or_none(item.find("district")),
            sponsorship_date=get_text_or_none(item.find("sponsorshipDate")),
            is_original_cosponsor=get_text_or_none(item.find("isOriginalCosponsor")),
        )
        result.append(cosponsor)
    return result


class CboCostEstimate(BaseModel):
    pub_date: datetime.datetime
    title: str
    url: str
    description: Optional[str]


def get_cbo_cost_estimates(element: Element) -> list[CboCostEstimate]:
    result = []
    if element is None:
        return result
    assert element.tag == "cboCostEstimates"
    for item in element:
        if item.tag != "item":
            continue
        cbo_cost_estimate = CboCostEstimate(
            pub_date=get_text_or_none(item.find("pubDate")),
            title=get_text_or_none(item.find("title")),
            url=get_text_or_none(item.find("url")),
            description=get_text_or_none(item.find("description")),
        )
        result.append(cbo_cost_estimate)
    return result


class Law(BaseModel):
    type: str
    number: str


def get_laws(element: Element) -> list[CboCostEstimate]:
    result = []
    if element is None:
        return result
    assert element.tag == "laws"
    for item in element:
        if item.tag != "item":
            continue
        law = Law(
            type=get_text_or_none(item.find("type")),
            number=get_text_or_none(item.find("number")),
        )
        result.append(law)
    return result


class Link(BaseModel):
    name: str
    url: str


def get_links(element: Element) -> list[Link]:
    result = []
    if element is None:
        return result
    assert element.tag == "links"
    for item in element:
        if item.tag != "link":
            continue
        link = Link(
            name=get_text_or_none(item.find("name")),
            url=get_text_or_none(item.find("url")),
        )
        result.append(link)
    return result


class Note(BaseModel):
    text: str
    links: list[Link]


def get_notes(element: Element) -> list[Note]:
    result = []
    if element is None:
        return result
    assert element.tag == "notes"
    for item in element:
        if item.tag != "item":
            continue
        note = Note(
            text=get_text_or_none(item.find("text")),
            links=get_links(item.find("links")),
        )
        result.append(note)
    return result


class AmendedBill(BaseModel):
    congress: int
    type: str
    origin_chamber: str
    origin_chamber_code: str
    number: int
    title: str
    update_date_including_text: datetime.datetime


def get_amended_bill(element: Element) -> Optional[AmendedBill]:
    if element is None:
        return None
    else:
        assert element.tag == "amendedBill"
        return AmendedBill(
            congress=element.find("congress").text,
            type=get_text_or_none(element.find("type")),
            origin_chamber=element.find("originChamber").text,
            origin_chamber_code=get_text_or_none(element.find("originChamberCode")),
            number=get_text_or_none(element.find("number")),
            title=element.find("title").text,
            update_date_including_text=get_text_or_none(
                element.find("updateDateIncludingText")
            ),
        )


class RecordedVote(BaseModel): 
    roll_number: int
    chamber: str
    congress: int
    date: datetime.datetime
    session_number: int
    url: str


def get_recorded_votes(element: Element) -> list[RecordedVote]:
    result = []
    if element is None:
        return result
    assert element.tag == "recordedVotes"
    for item in element:
        if item.tag != "recordedVote":
            continue
        recorded_vote = RecordedVote(
            roll_number=get_text_or_none(item.find("rollNumber")),
            chamber=get_text_or_none(item.find("chamber")),
            congress=get_text_or_none(item.find("congress")),
            date=get_text_or_none(item.find("date")),
            session_number=get_text_or_none(item.find("sessionNumber")),
            url = get_text_or_none(item.find("url")),
        )
        result.append(recorded_vote)
    return result


class ActionAmendment(BaseModel):
    action_date: datetime.date
    text: Optional[str]
    action_time: Optional[str]=None
    links: list[Link]
    type: Optional[str]=None
    action_code: Optional[str]=None
    source_system: Optional[SourceSystem]=None
    recorded_votes: Optional[list[RecordedVote]]=[]


def get_latest_action(element: Element) -> ActionAmendment:
    if element is None:
        return None
    else:
        assert element.tag == "latestAction"
        return ActionAmendment(
            action_date=get_text_or_none(element.find("actionDate")),
            text=get_text_or_none(element.find("text")),
            action_time=get_text_or_none(element.find("actionTime")),
            links=get_links(element.find("links")),
        )


def get_actions_amendments(element_outer: Element) -> list[ActionAmendment]:
    result = []
    if element_outer is None:
        return result
    assert element_outer.tag == "actions"
    count = int(element_outer.find("count").text)
    element = element_outer.find("actions")
    if element is None:
        return result

    for item in element:
        if item.tag != "item":
            continue
        action_amendment = ActionAmendment(
            action_date=get_text_or_none(item.find("actionDate")),
            action_time=get_text_or_none(item.find("actionTime")),
            text=get_text_or_none(item.find("text")),
            type=get_text_or_none(item.find("type")),
            action_code=get_text_or_none(item.find("actionCode")),
            source_system = get_source_system(item.find("sourceSystem")),
            recorded_votes = get_recorded_votes(item.find("recordedVotes")),
            links=get_links(item.find("links")),

        )
        result.append(action_amendment)
    return result


class Amendment(BaseModel):
    number: int
    congress: int
    type: str
    description: Optional[str]
    purpose: Optional[str]
    update_date: datetime.datetime
    latest_action: Optional[ActionAmendment]
    sponsors: list[Sponsor]
    submitted_date: datetime.datetime
    chamber: str
    links: list[Link]
    actions: list[ActionAmendment]


def get_amendments(element: Element) -> list[Amendment]:
    result = []
    if element is None:
        return result
    assert element.tag == "amendments"
    for item in element:
        if item.tag != "amendment":
            continue
        amendment = Amendment(
            number=get_text_or_none(item.find("number")),
            congress=get_text_or_none(item.find("congress")),
            type=get_text_or_none(item.find("type")),
            description=get_text_or_none(item.find("description")),
            purpose=get_text_or_none(item.find("purpose")),
            update_date=get_text_or_none(item.find("updateDate")),
            latest_action = get_latest_action(item.find("latestAction")),
            sponsors = get_sponsors(item.find("sponsors")),
            submitted_date = get_text_or_none(item.find("submittedDate")),
            chamber = get_text_or_none(item.find("chamber")),
            amended_bill = get_amended_bill(item.find("amendedBill")),
            links = get_links(item.find("links")),
            actions = get_actions_amendments(item.find("actions")),
        )
        result.append(amendment)
    return result



class TextVersionBillStatus(BaseModel):
    type: str
    date: Optional[datetime.datetime] = None
    url: Optional[str] = None


def get_text_versions(element: Optional[Element]) -> list[TextVersionBillStatus]:
    result = []
    if element is None:
        return result
    assert element.tag == "textVersions"
    for item in element:
        if item.tag != "item":
            continue
        formats = get_formats(item.find("formats"))
        if formats is None:
            url = None
        else:
            urls = list(set([el.url for el in formats]))
            if len(urls) == 0:
                url = None
            elif len(urls) == 1:
                url = urls[0]
            elif len(urls) > 1:
                raise ValueError("len(urls)>1")

        tv_type = get_text_or_none(item.find("type"))
        tv_date = get_text_or_none(item.find("date"))

        if url is None and tv_type is None and tv_date is None:
            continue
        else:
            text_version = TextVersionBillStatus(
                type=tv_type,
                date=tv_date,
                url=url,
            )
            result.append(text_version)
    return result


class Format(BaseModel):
    url: str


def get_formats(element: Optional[Element]) -> list[Format]:
    result = []
    if element is None:
        return result
    assert element.tag == "formats"
    for item in element:
        if item.tag != "item":
            continue
        el = Format(url=get_text_or_none(item.find("url")))
        result.append(el)
    return result


class Summary(BaseModel):
    version_code: str
    action_date: datetime.date
    action_desc: str
    update_date: datetime.datetime
    text: str


def get_summaries(element: Optional[Element]):
    result = []
    if element is None:
        return result
    assert element.tag == "summaries"
    for item in element:
        if item.tag != "summary":
            continue
        summary = Summary(
            version_code=get_text_or_none(item.find("versionCode")),
            action_date=get_text_or_none(item.find("actionDate")),
            action_desc=get_text_or_none(item.find("actionDesc")),
            update_date=get_text_or_none(item.find("updateDate")),
            text=get_text_or_none(item.find("text")),
        )
        result.append(summary)
    return result


class DublinCoreBillStatus(BaseModel):
    dc_format: str
    dc_language: str
    dc_rights: str
    dc_contributor: str
    dc_description: str


def get_dublin_core(element: Element):
    assert element.tag == "dublinCore"
    ns = {"dc": "http://purl.org/dc/elements/1.1/"}
    return DublinCoreBillStatus(
        dc_format=element.find("dc:format", namespaces=ns).text,
        dc_language=element.find("dc:language", namespaces=ns).text,
        dc_rights=element.find("dc:rights", namespaces=ns).text,
        dc_contributor=element.find("dc:contributor", namespaces=ns).text,
        dc_description=element.find("dc:description", namespaces=ns).text,
    )


def get_number(bill: Element) -> int:
    assert bill.tag == "bill"
    number = bill.find("number")
    bill_number = bill.find("billNumber")

    if number is None and bill_number is None:
        raise ValueError()
    elif number is None and bill_number is not None:
        return int(bill_number.text)
    elif number is not None and bill_number is None:
        return int(number.text)
    else:
        if number.text != bill_number.text:
            raise ValueError()
        return int(number.text)


def get_type(bill: Element) -> str:
    assert bill.tag == "bill"
    btype = bill.find("type")
    bill_type = bill.find("billType")

    if btype is None and bill_type is None:
        raise ValueError()
    elif btype is None and bill_type is not None:
        return bill_type.text
    elif btype is not None and bill_type is None:
        return btype.text
    else:
        if btype.text != bill_type.text:
            raise ValueError()
        return btype.text


def get_legislative_subjects(element: Optional[Element]) -> list[str]:
    result = []
    if element is None:
        return result
    assert element.tag == "legislativeSubjects"
    for item in element:
        if item.tag != "item":
            continue
        result.append(get_text_or_none(item.find("name")))
    return result


def get_subjects(element: Optional[Element]) -> list[str]:
    result = []
    if element is None:
        return result
    assert element.tag == "subjects"
    result = get_legislative_subjects(element.find("legislativeSubjects"))
    return result


def get_policy_area(element: Optional[Element]) -> Optional[str]:
    if element is None:
        return None
    assert element.tag == "policyArea"
    return get_text_or_none(element.find("name"))


class BillStatus(BaseModel):
    version: Optional[str] = None
    number: int
    update_date: datetime.datetime
    update_date_including_text: Optional[datetime.datetime] = None
    origin_chamber: str
    origin_chamber_code: Optional[str] = None
    type: str
    introduced_date: datetime.date
    congress: int
    committees: list[Committee]
    committee_reports: list[CommitteeReport]
    related_bills: list[RelatedBill]
    actions: list[Action]
    sponsors: list[Sponsor]
    cosponsors: list[Cosponsor]
    laws: list[Law]
    notes: list[Note]
    cbo_cost_estimates: list[CboCostEstimate]
    policy_area: Optional[str] = None
    subjects: list[str]
    summaries: list[Summary] = []
    title: str
    titles: list[Title]
    amendments: list[Amendment]
    text_versions: list[TextVersionBillStatus]
    latest_action: Action
    dublin_core: DublinCoreBillStatus

    @classmethod
    def from_xml_str(cls, xml: str):

        root = ET.fromstring(xml)
        version = root.find("version")
        bill = root.find("bill")

        return cls(
            version=get_text_or_none(version),
            number=get_number(bill),
            update_date=bill.find("updateDate").text,
            update_date_including_text=get_text_or_none(
                bill.find("updateDateIncludingText")
            ),
            origin_chamber=bill.find("originChamber").text,
            origin_chamber_code=get_text_or_none(bill.find("originChamberCode")),
            type=get_type(bill),
            introduced_date=bill.find("introducedDate").text,
            congress=bill.find("congress").text,
            committees=get_committees(bill.find("committees")),
            committee_reports=get_committee_reports(bill.find("committeeReports")),
            related_bills=get_related_bills(bill.find("relatedBills")),
            actions=get_actions(bill.find("actions")),
            sponsors=get_sponsors(bill.find("sponsors")),
            cosponsors=get_cosponsors(bill.find("cosponsors")),
            laws=get_laws(bill.find("laws")),
            notes=get_notes(bill.find("notes")),
            cbo_cost_estimates=get_cbo_cost_estimates(bill.find("cboCostEstimates")),
            policy_area=get_policy_area(bill.find("policyArea")),
            subjects=get_subjects(bill.find("subjects")),
            summaries=get_summaries(bill.find("summaries")),
            title=bill.find("title").text,
            titles=get_titles(bill.find("titles")),
            amendments=get_amendments(bill.find("amendments")),
            text_versions=get_text_versions(bill.find("textVersions")),
            latest_action=get_action(bill.find("latestAction")),
            dublin_core=get_dublin_core(root.find("dublinCore")),
        )


if __name__ == "__main__":

    import pandas as pd
    import rich
    from collections import Counter

    congress_hf_path = Path("/Users/galtay/data/congress-hf")
    xml_file_path = congress_hf_path / "usc-109-billstatus.parquet"
    df = pd.read_parquet(xml_file_path)
    tags = Counter()
    for _, row in df.iterrows():
        xml = row["xml"]
        bs = BillStatus.from_xml_str(xml)

        bill = ET.fromstring(xml).find("bill")
        for element in bill:
            tags[element.tag] += 1
#            if element.tag == "amendments":
#                sys.exit(0)


