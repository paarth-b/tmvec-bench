"""Shared sequence ID cleaning utilities."""


def clean_seq_id_column(series):
    """Remove CATH prefixes ('cath|4_4_0|') and range suffixes ('/1-150') from a Series of IDs."""
    return series.str.split('/').str[0].str.split('|').str[-1]
