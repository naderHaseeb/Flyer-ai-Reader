from __future__ import annotations

import re
from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def validate_iso_date(value: str | None) -> str | None:
    if value is None:
        return None
    if not DATE_PATTERN.fullmatch(value):
        raise ValueError("date must use YYYY-MM-DD")
    date.fromisoformat(value)
    return value


class FlyerContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shop_name: str | None = None
    campaign_name: str | None = None
    flyer_start_date: str | None = None
    flyer_end_date: str | None = None
    region: str | None = None
    branch: str | None = None
    currency: str | None = None

    _validate_dates = field_validator(
        "flyer_start_date",
        "flyer_end_date",
    )(validate_iso_date)


class QwenProduct(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    id: int = Field(gt=0, strict=True)
    name: str | None = Field(default=None, alias="product_name")
    quantity: str | None = None
    price_before: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    price_after: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    currency: str | None = None
    start_date: str | None = Field(default=None, alias="product_start_date")
    end_date: str | None = Field(default=None, alias="product_end_date")
    date_source: str | None = None
    date_badge_text: str | None = None

    _validate_dates = field_validator("start_date", "end_date")(
        validate_iso_date
    )


class SectionDate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str | None = None
    date: str
    top: float = Field(ge=0, le=1000, allow_inf_nan=False)
    bottom: float = Field(ge=0, le=1000, allow_inf_nan=False)

    _validate_date = field_validator("date")(validate_iso_date)

    @model_validator(mode="after")
    def validate_bounds(self):
        if self.bottom <= self.top:
            raise ValueError("section bottom must be greater than top")
        return self
