import re
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Iterable
import pandas_flavor as pf

from statsmodels.iolib.summary2 import summary_col
import pyperclip

try:
    from janitor.utils import skiperror, skipna
except ImportError:

    def skipna(f: Callable) -> Callable:
        """
        Decorator for escaping np.nan and None in a function

        Should be used like this::

            df[column].apply(skipna(transform))

        or::

            @skipna
            def transform(x):
                pass

        :param f: the function to be wrapped
        :returns: _wrapped, the wrapped function
        """

        def _wrapped(x, *args, **kwargs):
            if (type(x) is float and np.isnan(x)) or x is None:
                return np.nan
            else:
                return f(x, *args, **kwargs)

        return _wrapped

    def skiperror(f: Callable, return_x: bool = False, return_val=np.nan) -> Callable:
        """
        Decorator for escaping errors in a function

        Should be used like this::

            df[column].apply(
                skiperror(transform, return_val=3, return_x=False))

        or::

            @skiperror(return_val=3, return_x=False)
            def transform(x):
                pass

        :param f: the function to be wrapped
        :param return_x: whether or not the original value that caused error
            should be returned
        :param return_val: the value to be returned when an error hits.
            Ignored if return_x is True
        :returns: _wrapped, the wrapped function
        """

        def _wrapped(x, *args, **kwargs):
            try:
                return f(x, *args, **kwargs)
            except Exception:
                if return_x:
                    return x
                return return_val

        return _wrapped


@dataclass
class Enclosure:
    begin: str
    end: str
    begin_regex: str
    end_regex: str
    middle_regex: str = r"."
    inner_enclosure: object = None
    strip_math: bool = False

    def strip_string(self, string: str) -> str:
        """Strip a string according to the patterns defined"""
        re_pattern = (
            self.begin_regex + r"(" + self.middle_regex + r"*?)" + self.end_regex
        )
        inner = (re.findall(re_pattern, string) + [None])[0]
        if self.inner_enclosure is not None:
            return self.inner_enclosure.strip_string(inner)
        return inner

    def _strip_math(self, string):
        if string[0] == "$" and string[-1] == "$":
            return string[1:-1]
        else:
            return string

    def construct_string(self, string):
        """Reconstruct a string according to the patterns defined"""
        if self.inner_enclosure is not None:
            s = self.inner_enclosure.construct_string(string)
        else:
            s = string

        if self.strip_math:
            s = self._strip_math(s)
        result = f"{self.begin}{s}{self.end}"

        if self.strip_math:
            result = f"${result}$"
        return result


# Canned enclosures
PARENTHESIS = Enclosure("(", ")", r"^\(", r"\)$")
SQUARE_BRACKET = Enclosure("[", "]", r"^\[", r"\]$")
CURLY_BRACKET = Enclosure("\{", "\}", r"^{", r"}$")
DOLLAR_PREFIX = Enclosure("\$", "", r"^\$", "$", middle_regex="[^\$]", strip_math=True)
PERCENT_SUFFIX = Enclosure("", "\%", r"^", r"\%$", strip_math=True)

ENCLOSURES = [PARENTHESIS, SQUARE_BRACKET, CURLY_BRACKET, DOLLAR_PREFIX, PERCENT_SUFFIX]


kwarg_docstring = """\
num_digit : int, optional
        Number of digits after the decimal point represented, by default 3.
        Integers that are > 1 are ignored.
    large : int, optional
        The upper limit until scientific notation; scientific notation is activated if
        s > 10 ** large; the parameter is ignored if auto_scientific is False, by default 9
    na_rep : str, optional
        Representation of missing values (NaN, nan, None, etc.), by default "---"
    additional_enclosures : Iterable[Enclosure], optional
        Additional enclosures to be checked by mathify, by default []
    auto_scientific : bool, optional
        Whether to convert automatically into scientific notation, by default True.
        Scientific notation is activated if number is outside of
        [10 ** -num_digit, 10 ** large]
"""


def mathify(
    s,
    num_digit: int = 3,
    large: int = 9,
    na_rep: str = "---",
    additional_enclosures: Iterable[Enclosure] = [],
    auto_scientific: bool = True,
) -> str:
    """
    Convert a string or a numeric variable into LaTeX representations.

    - Automatically adds commas to large integers
    - Can automatically turn string into scientific notation
    - Automatically parses enclosures

    Parameters
    ----------
    s : str or float or int
        The string to be converted into LaTeX
    num_digit : int, optional
        Number of digits after the decimal point represented, by default 3.
        Integers that are > 1 are ignored.
    large : int, optional
        The upper limit until scientific notation; scientific notation is activated if
        s > 10 ** large; the parameter is ignored if auto_scientific is False, by default 9
    na_rep : str, optional
        Representation of missing values (NaN, nan, None, etc.), by default "---"
    additional_enclosures : Iterable[Enclosure], optional
        Additional enclosures to be checked by mathify, by default []
    auto_scientific : bool, optional
        Whether to convert automatically into scientific notation, by default True.
        Scientific notation is activated if number is outside of
        [10 ** -num_digit, 10 ** large]

    Returns
    -------
    str
        String converted into LaTeX
    """

    if (
        (type(s) is float and np.isnan(s))
        or (type(s) is str and s.lower() == "nan")
        or s is None
    ):
        return na_rep

    # Handle trailing stars
    stars = ""
    n_stars = 0
    if type(s) is str and s[-1] == "*":
        stars = re.findall(r"\*+", s)[0]
        n_stars = len(stars)
        s = float(re.sub(r"\*+", "", s))

    # Escape enclosures
    if type(s) is str:
        for enc in ENCLOSURES + additional_enclosures:
            try:
                inner = enc.strip_string(s)
                if inner is None:
                    continue

                mathified_inner = mathify(
                    inner,
                    num_digit=num_digit,
                    large=large,
                    auto_scientific=auto_scientific,
                )

                s = enc.construct_string(mathified_inner)
                return f"{s}\\textsuperscript{{{stars}}}" if n_stars > 0 else s
            except Exception:
                continue

    s = float(s)

    # Scientific notation and automatic comma in integer formatting
    if (
        (abs(s) / (10 ** large) > 1 or abs(s) < 10 ** -num_digit)
        and s != 0
        and auto_scientific
    ):
        s = (f"%.{num_digit}E") % s
        lst = s.split("E")
        s = f"{lst[0]} \\times 10^{{{int(lst[-1])}}}"
    elif abs(s - round(s)) < 1e-10 and s > 1:
        s = "{:,}".format(int(round(s)))
    else:
        s = ("{:." + str(num_digit) + "f}").format(s)

    s = f"${s}$"

    return f"{s}\\textsuperscript{{{stars}}}" if n_stars > 0 else s


def mathify_column(x: pd.Series, **kwargs) -> pd.Series:
    """
    Mathify (when one could) a pd.Series iteratively

    Parameters
    ----------
    x : pd.Series
        The input series

    **kwargs:
        num_digit : int, optional
            Number of digits after the decimal point represented, by default 3.
            Integers that are > 1 are ignored.
        large : int, optional
            The upper limit until scientific notation; scientific notation is activated if
            s > 10 ** large; the parameter is ignored if auto_scientific is False, by default 9
        na_rep : str, optional
            Representation of missing values (NaN, nan, None, etc.), by default "---"
        additional_enclosures : Iterable[Enclosure], optional
            Additional enclosures to be checked by mathify, by default []
        auto_scientific : bool, optional
            Whether to convert automatically into scientific notation, by default True.
            Scientific notation is activated if number is outside of
            [10 ** -num_digit, 10 ** large]

    Returns
    -------
    pd.Series
        The output series
    """
    return x.apply(skiperror(lambda y: mathify(y, **kwargs), return_x=True))


@pf.register_dataframe_method
def mathify_table(
    df: pd.DataFrame, texttt_index: bool = False, texttt_column: bool = False, **kwargs
) -> pd.DataFrame:
    """
    Mathify a pd.DataFrame iteratively

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame
    texttt_index : bool, optional
        Whether or not to escape index and convert index labels to \\texttt,
        by default False
    texttt_column : bool, optional
        Whether or not to escape index and convert column labels to \\texttt,
        by default False


    **kwargs:
        num_digit : int, optional
            Number of digits after the decimal point represented, by default 3.
            Integers that are > 1 are ignored.
        large : int, optional
            The upper limit until scientific notation; scientific notation is activated if
            s > 10 ** large; the parameter is ignored if auto_scientific is False, by default 9
        na_rep : str, optional
            Representation of missing values (NaN, nan, None, etc.), by default "---"
        additional_enclosures : Iterable[Enclosure], optional
            Additional enclosures to be checked by mathify, by default []
        auto_scientific : bool, optional
            Whether to convert automatically into scientific notation, by default True.
            Scientific notation is activated if number is outside of
            [10 ** -num_digit, 10 ** large]

    Returns
    -------
    d : pd.DataFrame
        The output DataFrame
    """

    d = df.apply(skiperror(lambda x: mathify_column(x, **kwargs), return_x=True))
    if texttt_index:
        d.index = [
            "\\texttt{{{}}}".format(s.replace("_", "\\_"))
            if (len(s) > 0 and s[0] != "$" and s[-1] != "$")
            else s
            for s in d.index.astype(str)
        ]
    if texttt_column:
        d.columns = [
            "\\texttt{{{}}}".format(s.replace("_", "\\_"))
            if (len(s) > 0 and s[0] != "$" and s[-1] != "$")
            else s
            for s in np.array(d.columns).astype(str)
        ]
    return d


def interleave(*args: Iterable) -> Iterable:
    """[x1,x2,x3], [y1,y2,y3], [z1,z2,z3] -> (x1,y1,z1), ..., (x3,y3,z3)"""
    return [val for seq in zip(*args) for val in seq]


@pf.register_dataframe_method
def to_latex_table(
    df: pd.DataFrame,
    caption: str,
    mathify_first: bool = True,
    label: str = None,
    filename: str = None,
    insert_column_number: bool = True,
    notes: str = None,
    star_notes: str = None,
    vspace: float = 1,
    additional_text: str = "",
    mathify_args: dict = dict(),
    to_latex_args: dict = None,
) -> str:
    """
    Convert a pd.DataFrame to string for LaTeX table or threeparttable environment

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    caption : str
        \caption field for the table
    mathify_first : bool, optional
        Whether or not the table is passed through mathify first, by default True
    label : str, optional
        \label field for the table, if not specified, then a underscore case of
        the caption is provided, by default None
    filename : str, optional
        File name to save the table in if specified,
        code to input the table is copied to the clipboard, by default None
    insert_column_number : bool, optional
        Number columns (1), (2), etc., by default True
    notes : str, optional
        Additional notes in the table, by default None
    star_notes : str, optional
        Additional notes in the table regarding significance stars, by default None
    vspace : float, optional
        How much vspace (in em terms) is between caption and table, by default 1
    additional_text : str, optional
        Additional commands added after caption field, by default ""
    mathify_args : dict, optional
        Arguments passed to mathify, by default dict()
        Arguments:
            texttt_column : bool
            texttt_index : bool
            num_digit : int, optional
                Number of digits after the decimal point represented, by default 3.
                Integers that are > 1 are ignored.
            large : int, optional
                The upper limit until scientific notation; scientific notation is activated if
                s > 10 ** large; the parameter is ignored if auto_scientific is False, by default 9
            na_rep : str, optional
                Representation of missing values (NaN, nan, None, etc.), by default "---"
            additional_enclosures : Iterable[Enclosure], optional
                Additional enclosures to be checked by mathify, by default []
            auto_scientific : bool, optional
                Whether to convert automatically into scientific notation, by default True.
                Scientific notation is activated if number is outside of
                [10 ** -num_digit, 10 ** large]
    to_latex_args : dict, optional
        Arguments passed to pd.DataFrame.to_latex, by default dict()
        See pd.DataFrame.to_latex

    Returns
    -------
    str
        LaTeX code for the table
    """

    if not label:
        label = "_".join(map(lambda s: re.sub(r"\W+", "", s), caption.lower().split()))

    if mathify_first:
        d = df.copy()
        t = mathify_table(d, **mathify_args)
    else:
        t = df.copy()

    if insert_column_number:
        t = t.T.set_index(
            np.array(
                [
                    f"\hypertarget{{tabcol:{label + str(r)}}}{{({r})}}"
                    for r in range(1, t.shape[-1] + 1)
                ]
            ),
            append=True,
        ).T.copy()

    if to_latex_args is None:
        to_latex_args = {}

    if "column_format" not in to_latex_args:
        to_latex_args["column_format"] = "l" + "c" * df.shape[-1]

    opt_val = pd.get_option("max_colwidth")
    pd.set_option("max_colwidth", 10000)
    table_str = t.to_latex(escape=False, **to_latex_args)
    pd.set_option("max_colwidth", opt_val)

    # No threeparttable
    if notes is None and star_notes is None:
        s = f"""\\begin{{table}}[tbh]
        \\caption{{{caption}}}
        \\label{{tab:{label}}}
        \\centering
        \\vspace{{1em}}
        {table_str}
        \\end{{table}}"""

    # Threeparttable
    else:
        notes = (
            f"{star_notes if star_notes is not None else ''}"
            f"{notes if notes is not None else ''}"
        )

        s = f"""
        \\begin{{table}}[tbh]
        \\caption{{{caption}}}
        \\label{{tab:{label}}}
        {additional_text}
        \\centering
        \\vspace{{{vspace}em}}
        \\begin{{threeparttable}}
        {table_str}
        \\begin{{tablenotes}}
        \\footnotesize
        \item {notes}
        \end{{tablenotes}}
        \end{{threeparttable}}

        \end{{table}}
        """

    if filename:
        with open(filename, "w") as f:
            f.write(s)
        pyperclip.copy(f"\\input{{{filename}}}  % \\label{{tab:{label}}}")
    return s


@pf.register_dataframe_method
def consolidate_se(
    df: pd.DataFrame,
    coef_cols: Iterable,
    *se_cols: Iterable,
    add_stars: bool = False,
    thresh: Iterable = [0.001, 0.01, 0.05],
) -> pd.DataFrame:
    """
    Collapse columns to represent standard errors in regression-table style. Standard errors
    are in parentheses, brackets, and curly brackets. Additional standard errors are in
    curly brackets

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    coef_cols : Iterable
        List of column names for coefficient
    *se_cols : Iterable
        List of list of standard error names for coefficient by type
        example: ses_type1, ses_type2, ...
    add_stars : bool, optional
        Add stars to coefficients, by default False
        Assuming distribution is Normal under the null hypothesis
    thresh : Iterable, optional
        If add_stars is True, thresholds for increasing stars,
        by default [0.001, 0.01, 0.05]

    Returns
    -------
    pd.DataFrame
        Output DataFrame
    """

    from scipy.stats import norm

    # Rest of the columns
    rest = list(
        c
        for c in df.columns
        if (c not in coef_cols and np.array([c not in se_c for se_c in se_cols]).all())
    )

    return_df = []
    for seq in zip(coef_cols, *se_cols):
        coef_col = seq[0]
        se_col = seq[1]
        p_vals = norm.cdf(-np.abs(df[coef_col] / df[se_col]).values) * 2

        coef = df[coef_col].copy()
        if add_stars:
            for i, p in enumerate(p_vals):
                if p < thresh[0]:
                    coef.iloc[i] = str(coef.values[i]) + "***"
                elif p < thresh[1]:
                    coef.iloc[i] = str(coef.values[i]) + "**"
                elif p < thresh[2]:
                    coef.iloc[i] = str(coef.values[i]) + "*"
                else:
                    coef.iloc[i] = str(coef.values[i])
        ses = [
            df[s].apply(
                skipna(
                    lambda x: ENCLOSURES[:3][i].construct_string(str(x))
                    if i < 3
                    else ENCLOSURES[2].construct_string(str(x))
                )
            )
            for i, s in enumerate(seq[1:])
        ]

        v = interleave(coef.values, *ses)
        return_df.append(
            pd.Series(
                v,
                index=interleave(coef.index, *([""] * len(se) for se in ses)),
                name=coef_col,
            )
        )

    # Add the rest of the columns to return_df
    for coef_col in rest:
        coef = df[coef_col].copy()
        v = interleave(coef.values, *([[""] * len(coef)] * len(se_cols)))
        return_df.append(
            pd.Series(
                v,
                index=interleave(coef.index, *([[""] * len(coef)] * len(se_cols))),
                name=coef_col,
            )
        )
    col_order = list(filter(lambda x: x in coef_cols or x in rest, df.columns))
    return pd.concat(return_df, axis=1, sort=False)[col_order].copy()


def regression_table(regs, notes=None, stars=True, **kwargs):
    """
    Create a pandas.DataFrame object summarizing a series of regressions

    Parameters
    ----------
    regs: a list of statsmodels.regression.linear_model.RegressionResults
        objects, one for each column of the regression table

    notes: optional, a dict of additional rows to the table. Each key (string) is
        the name of a row, and each associated value (list of string) is the content


    stars : bool, optional
        whether or not to include stars, by default False

    **kwargs: Additional arguments passed to statsmodels.iolib.summary2.summary_col
        Summarize multiple results instances side-by-side (coefs and SEs)

        results : statsmodels results instance or list of result instances
        float_format : string, optional
            float format for coefficients and standard errors
            Default : '%.4f'
        model_names : list of strings, optional
            Must have same length as the number of results. If the names are not
            unique, a roman number will be appended to all model names
        stars : bool
            print significance stars
        info_dict : dict
            dict of functions to be applied to results instances to retrieve
            model info. To use specific information for different models, add a
            (nested) info_dict with model name as the key.
            Example: `info_dict = {"N":..., "R2": ..., "OLS":{"R2":...}}` would
            only show `R2` for OLS regression models, but additionally `N` for
            all other results.
            Default : None (use the info_dict specified in
            result.default_model_infos, if this property exists)
        regressor_order : list of strings, optional
            list of names of the regressors in the desired order. All regressors
            not specified will be appended to the end of the list.
        drop_omitted : bool, optional
            Includes regressors that are not specified in regressor_order. If False,
            regressors not specified will be appended to end of the list. If True,
            only regressors in regressors_list will be included.

    Returns
    -------
    t: pd.DataFrame, a pandas.DataFrame of a regression table
    """

    d = {
        "$N$": lambda x: "{0:d}".format(int(x.nobs)),
        "$R^2$": lambda x: "{:.2f}".format(x.rsquared),
    }
    t = summary_col(regs, stars=stars, info_dict=d, **kwargs).tables[0].copy()
    if notes:
        for k, v in notes.items():
            t.loc[k] = v
    return t
