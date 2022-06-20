import pandas as pd
import numpy as np


def parse_excel(
    filename: str, sheet: str = 0, skiprows: int = 0, header: int = None
) -> pd.DataFrame:
    """
    Parse an excel sheet skipping some rows.

    Can read both xlsx and csv files. No header is assumed.

    Parameters
    ----------
    filename: str
        The name of the file to parse.
    sheet: str
        The name of the sheet to parse.
        If empty, the standard sheet is parsed.
    skiprows: int
        The number of rows to skip to parse the file.
        If empty, assumed 0 rows to skip
    header: int
        The index to use as header.
        If there is no header (default), None.

    Returns
    -------
    pandas.DataFrame
        The table in the excel file as a pandas.DataFrame.
    """
    if not filename:
        raise FileNotFoundError("No file given!")

    if filename.endswith(".csv"):
        return (
            pd.read_csv(filename, skiprows=skiprows, header=header)
            .dropna(how="all")
            .dropna(axis=1, how="all")
        )
    elif filename.endswith(".xlsx"):
        return (
            pd.read_excel(
                filename,
                engine="openpyxl",
                sheet_name=sheet,
                skiprows=skiprows,
                header=header,
            )
            .dropna(how="all")
            .dropna(axis=1, how="all")
        )


def get_keywords(
    filename: str,
    keyword_column_name: str,
    sheet: str = 0,
    skiprows: int = 0,
    header: int = None,
):
    if not keyword_column_name:
        raise RuntimeError(
            "Keyword column not selected! Please five an index or the column name."
        )
    df = parse_excel(filename, sheet, skiprows, header)
    if isinstance(keyword_column_name, str):
        return df.loc[:, keyword_column_name].tolist()
    elif isinstance(keyword_column_name, int):
        return df.iloc[:, keyword_column_name].tolist()
    else:
        raise TypeError("Keyword indication is neither a string nor an int!")
