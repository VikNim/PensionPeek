from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


def _as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(str(value).replace(",", "")))
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True, slots=True)
class Filing:
    filing_id: str
    plan_name: str
    sponsor: str
    ein: str
    plan_number: str
    plan_year: int | None
    city: str = ""
    state: str = ""
    date_received: str = ""
    participants_boy: int | None = None
    assets_eoy: float | None = None
    pdf_path: str = ""
    dcg_indicator: bool = False

    @classmethod
    def from_hit(cls, hit: dict[str, Any]) -> Filing:
        fields = hit.get("fields", {})
        return cls(
            filing_id=str(hit.get("id", "")),
            plan_name=str(fields.get("planname", "Unknown plan")),
            sponsor=str(fields.get("plansponsor", "Unknown sponsor")),
            ein=str(fields.get("ein", "")),
            plan_number=str(fields.get("pn", "")),
            plan_year=_as_int(fields.get("planyear")),
            city=str(fields.get("city", "")),
            state=str(fields.get("state", "")),
            date_received=str(fields.get("datereceived", "")),
            participants_boy=_as_int(fields.get("participantsboy")),
            assets_eoy=_as_float(fields.get("assetseoy")),
            pdf_path=str(fields.get("pdfpath", "")),
            dcg_indicator=str(fields.get("dcgind", "0")) == "1",
        )

    @property
    def key(self) -> str:
        return self.filing_id or f"{self.ein}-{self.plan_number}-{self.plan_year}"

    @property
    def formatted_ein(self) -> str:
        digits = "".join(char for char in self.ein if char.isdigit())
        return f"{digits[:2]}-{digits[2:]}" if len(digits) == 9 else self.ein

    @property
    def received_date(self) -> str:
        if not self.date_received:
            return "Not reported"
        try:
            return datetime.fromisoformat(self.date_received.replace("Z", "+00:00")).strftime(
                "%b %d, %Y"
            )
        except ValueError:
            return self.date_received.split("T")[0]

    @property
    def location(self) -> str:
        return ", ".join(part for part in (self.city, self.state) if part) or "Not reported"


@dataclass(frozen=True, slots=True)
class TextChunk:
    chunk_id: str
    page: int
    text: str


@dataclass(frozen=True, slots=True)
class AssetHolding:
    """One line item from a Form 5500 Schedule of Assets (Held at End of Year).

    `identity` is the combined issuer + fund/investment description text as printed
    (e.g. "The Vanguard Group, Inc. 500 Index Fund Institutional Select") -- the two
    aren't split, because PDF text extraction gives no reliable delimiter between them
    and classification only needs the combined text anyway.
    """

    identity: str
    value: float


@dataclass(slots=True)
class FilingMetrics:
    assets_boy: float | None = None
    assets_eoy: float | None = None
    liabilities_boy: float | None = None
    liabilities_eoy: float | None = None
    net_assets_boy: float | None = None
    net_assets_eoy: float | None = None
    total_income: float | None = None
    total_expenses: float | None = None
    participants_boy: int | None = None
    participants_eoy: int | None = None
    asset_categories: dict[str, float] = field(default_factory=dict)
    holdings: list[AssetHolding] = field(default_factory=list)

    @property
    def net_income(self) -> float | None:
        if self.total_income is None or self.total_expenses is None:
            return None
        return self.total_income - self.total_expenses


@dataclass(slots=True)
class ParsedFiling:
    pages: list[str]
    chunks: list[TextChunk]
    metrics: FilingMetrics
    warnings: list[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(self.pages)

    @property
    def character_count(self) -> int:
        return sum(len(page) for page in self.pages)


@dataclass(frozen=True, slots=True)
class RetrievedFiling:
    pdf_bytes: bytes
    filename: str
    source_url: str
    was_archive: bool = False


@dataclass(frozen=True, slots=True)
class VulFiling:
    """One SEC EDGAR N-6 (or N-6/A) filing for a variable life separate account."""

    cik: str
    accession_number: str
    form_type: str
    file_date: str
    display_name: str
    document_filename: str

    @property
    def key(self) -> str:
        return self.accession_number

    @property
    def accession_no_dashes(self) -> str:
        return self.accession_number.replace("-", "")


@dataclass(frozen=True, slots=True)
class RetrievedVulDocument:
    html: str
    filename: str
    source_url: str


@dataclass(frozen=True, slots=True)
class SubFund:
    name: str
    manager: str | None
    allocation_weight: float | None
    expense_ratio: float | None
    source_section: str | None = None


@dataclass(frozen=True, slots=True)
class AllocationCategory:
    asset_category: str
    percentage: float
    source_section: str | None = None


@dataclass(frozen=True, slots=True)
class VulRiskMetrics:
    fixed_income_percentage: float | None = None
    equity_exposure_percentage: float | None = None
    guaranteed_floor_rate: float | None = None
    upside_cap_rate: float | None = None


@dataclass(frozen=True, slots=True)
class VulExtraction:
    """Structured output of the LLM extraction step over one VUL filing."""

    policy_investment_type: str
    carrier_name: str | None
    reporting_year: int | None
    allocation_chart: list[AllocationCategory] = field(default_factory=list)
    unclassified_percentage: float = 0.0
    unclassified_reason: str | None = None
    risk_metrics: VulRiskMetrics = field(default_factory=VulRiskMetrics)
    sub_funds_list: list[SubFund] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
