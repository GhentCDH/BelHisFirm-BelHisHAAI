import logging
import re

_log = logging.getLogger("index_parser")

_ADDR_PARTICLES = re.compile(r'\b(à|te)\b\s*')


class CRFPostProcessor:
    """
    Rules applied to the CRF output DataFrame before it is saved.

    - process(df) : return the cleaned DataFrame
    """

    def process(self, df):
        df = self._drop_delimiter(df)
        df = self._clean_address(df)
        df = self._drop_consecutive_duplicate_records(df)
        return df

    # --- column rules ---

    def _drop_delimiter(self, df):
        return df.drop(columns=["Delimiter"], errors="ignore")

    def _clean_address(self, df):
        if "Addres" not in df.columns:
            return df
        df["Addres"] = df["Addres"].apply(
            lambda v: _ADDR_PARTICLES.sub("", v).strip() if isinstance(v, str) else v
        )
        return df

    def _drop_consecutive_duplicate_records(self, df):
        if "RecordID" not in df.columns:
            return df
        # An empty RecordID next to another empty RecordID isn't a real duplicate — it
        # just means the CRF didn't find an id on either line — so only collapse
        # consecutive rows that share the same *non-empty* RecordID.
        non_empty = df["RecordID"].astype(str).str.strip().ne("")
        is_duplicate = non_empty & df["RecordID"].eq(df["RecordID"].shift(1))
        if is_duplicate.any():
            dropped = df.loc[is_duplicate, "RecordID"].tolist()
            _log.warning(f"{len(dropped)} rij(en) verwijderd als opeenvolgende dubbele RecordID: {dropped}")
        return df[~is_duplicate].reset_index(drop=True)
