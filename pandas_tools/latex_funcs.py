import re

def generate_enclosing_pattern(encloser):
    l = len(encloser)
    begin = encloser[:l // 2]
    end = encloser[l // 2:]
    re_pattern = begin + r'(.+?)' + end
    return re_pattern

PARENTHESIS_PATTERN = generate_enclosing_pattern(r"\(\)")
SQUARE_BRACKET_PATTERN = generate_enclosing_pattern(r"\[\]")
CURLY_BRACKET_PATTERN = generate_enclosing_pattern(r"{}")


def mathify(s, num_digit=3, large=9, phantom_space=False,
            max_stars=0, na_rep="---",
            auto_scientific=True, se=False):
    """
    Transform a float or string that contains a float into a string where the
    numerical part is wrapped in LaTeX math environment.
    """
    stars = ''
    n_stars = 0
    if (type(s) is float and np.isnan(s)) or s == "NaN":
        return na_rep
    if type(s) is str and s[-1] == "*":
        stars = re.findall(r'\*+', s)[0]
        n_stars = len(stars)
        s = float(re.sub(r'\*+', '', s))

    if type(s) is str and s[0] == "(" and s[-1] == ")":
        s = f"({mathify(float(s[1:-1]), num_digit=num_digit, large=large, max_stars=0, auto_scientific=auto_scientific)})"
    else:
        if type(s) is str:
            s = float(s)
        if (abs(s) / (10**large) > 1 or abs(s) < 10 ** -num_digit) and s != 0 and auto_scientific:
            s = ((f'%.{num_digit}E') % s)
            lst = s.split("E")
            s = f"{lst[0]} \\times 10^{{{int(lst[-1])}}}"
        elif abs(s - round(s)) < 1e-10:
            s = "{:,}".format(int(round(s)))

        else:
            s = ('{:.' + str(num_digit) + 'f}').format(s)
        s = f'${s}$'

    return f"{s}\\textsuperscript{{{stars}}}"
