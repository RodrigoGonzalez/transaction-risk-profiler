""" Enum for the dataset columns """
from __future__ import annotations

from enum import Enum


class DataColumnsEnum(str, Enum):
    """Enum for the dataset columns"""

    ACCT_TYPE = "acct_type"
    APPROX_PAYOUT_DATE = "approx_payout_date"
    BODY_LENGTH = "body_length"
    CHANNELS = "channels"
    COUNTRY = "country"
    CURRENCY = "currency"
    DELIVERY_METHOD = "delivery_method"
    DESCRIPTION = "description"
    EMAIL_DOMAIN = "email_domain"
    EVENT_CREATED = "event_created"
    EVENT_END = "event_end"
    EVENT_PUBLISHED = "event_published"
    EVENT_START = "event_start"
    FB_PUBLISHED = "fb_published"
    GTS = "gts"  # Gross Ticket Sales
    HAS_ANALYTICS = "has_analytics"
    HAS_HEADER = "has_header"
    HAS_LOGO = "has_logo"
    LISTED = "listed"
    NAME = "name"
    NAME_LENGTH = "name_length"
    NUM_ORDER = "num_order"
    NUM_PAYOUTS = "num_payouts"
    OBJECT_ID = "object_id"
    ORG_DESC = "org_desc"
    ORG_FACEBOOK = "org_facebook"
    ORG_NAME = "org_name"
    ORG_TWITTER = "org_twitter"
    PAYEE_NAME = "payee_name"
    PAYOUT_TYPE = "payout_type"
    PREVIOUS_PAYOUTS = "previous_payouts"
    SALE_DURATION = "sale_duration"
    SALE_DURATION2 = "sale_duration2"
    SHOW_MAP = "show_map"
    TICKET_TYPES = "ticket_types"
    USER_AGE = "user_age"
    USER_CREATED = "user_created"
    USER_TYPE = "user_type"
    VENUE_ADDRESS = "venue_address"
    VENUE_COUNTRY = "venue_country"
    VENUE_LATITUDE = "venue_latitude"
    VENUE_LONGITUDE = "venue_longitude"
    VENUE_NAME = "venue_name"
    VENUE_STATE = "venue_state"

    def __str__(self):
        return self.value

    @classmethod
    def get_target_column(cls) -> DataColumnsEnum:
        """
        Get the target column name defined in the Enum.

        Returns
        -------
        DataColumnsEnum
            The target column name.
        """
        return cls.ACCT_TYPE

    @classmethod
    def get_target(cls) -> str:
        """
        Get the target column name defined in the Enum.

        Returns
        -------
        DataColumnsEnum
            The target column name.
        """
        return cls.get_target_column().value

    @classmethod
    def get_all_columns(cls) -> set[str]:
        """
        Get all column names defined in the Enum.

        Returns
        -------
        set[str]
            A list of all columns.
        """
        return set(cls)

    @classmethod
    def get_column_names(cls) -> list[str]:
        """
        Get all column names defined in the Enum.

        Returns
        -------
        list[str]
            A list of all columns.
        """
        return sorted(str(column) for column in cls.get_all_columns())

    @classmethod
    def get_all_features(cls) -> set[str]:
        """
        Get all feature names defined in the Enum (Non-target columns).

        Returns
        -------
        set[str]
            A set of all features.
        """
        return {column for column in cls.get_all_columns() if column != cls.get_target_column()}

    @classmethod
    def get_feature_names(cls) -> list[str]:
        """
        Get all feature names defined in the Enum (Non-target columns).

        Returns
        -------
        set[str]
            A set of all feature names.
        """
        return sorted(str(feature) for feature in cls.get_all_features())


class TargetEnum(str, Enum):
    """Enum for the target categories"""

    # Premium related
    PREMIUM = "premium"

    # Fraudster related
    FRAUDSTER = "fraudster"
    FRAUDSTER_EVENT = "fraudster_event"
    FRAUDSTER_ATT = "fraudster_att"

    # Spammer related
    SPAMMER_LIMITED = "spammer_limited"
    SPAMMER_NOINVITE = "spammer_noinvite"
    SPAMMER_WEB = "spammer_web"
    SPAMMER = "spammer"
    SPAMMER_WARN = "spammer_warn"

    # Terms of Service related
    TOS_WARN = "tos_warn"
    TOS_LOCK = "tos_lock"

    # Locked
    LOCKED = "locked"

    def __str__(self):
        return self.value

    @staticmethod
    def fraud_list() -> list[str]:
        """Returns a list of fraudster categories."""
        fraud_list = [
            TargetEnum.FRAUDSTER.value,
            TargetEnum.FRAUDSTER_EVENT.value,
            TargetEnum.FRAUDSTER_ATT.value,
        ]
        fraud_list += TargetEnum.suspicious_list()
        return fraud_list

    @staticmethod
    def spammer_list() -> list[str]:
        """Returns a list of spammer categories."""
        return [
            TargetEnum.SPAMMER_LIMITED.value,
            TargetEnum.SPAMMER_NOINVITE.value,
            TargetEnum.SPAMMER_WEB.value,
            TargetEnum.SPAMMER.value,
            TargetEnum.SPAMMER_WARN.value,
        ]

    @staticmethod
    def tos_list() -> list[str]:
        """Returns a list of terms of service violation categories."""
        return [TargetEnum.TOS_WARN.value, TargetEnum.TOS_LOCK.value]

    @staticmethod
    def locked_list() -> list[str]:
        """Returns a list of locked categories."""
        return [TargetEnum.LOCKED.value]

    @staticmethod
    def suspicious_list() -> list[str]:
        """Returns a list of suspicious categories."""
        return TargetEnum.spammer_list() + TargetEnum.tos_list() + TargetEnum.locked_list()


class TicketTypesColumnsEnum(str, Enum):
    """Enum for the ticket_types column"""

    COST = "cost"
    AVAILABILITY = "availability"
    QUANTITY_TOTAL = "quantity_total"
    QUANTITY_SOLD = "quantity_sold"
    EVENT_ID = "event_id"

    def __str__(self):
        return self.value


class PreviousPayoutsColumnsEnum(str, Enum):
    """Enum for the previous_payouts column"""

    NAME = "name"
    AMOUNT = "amount"
    ADDRESS = "address"
    COUNTRY = "country"
    CREATED = "created"

    STATE = "state"
    UID = "uid"
    EVENT = "event"
    ZIP_CODE = "zip_code"

    def __str__(self):
        return self.value
