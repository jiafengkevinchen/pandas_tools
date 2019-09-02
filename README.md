# `pandas_tools`: auxiliary functionalities for pandas

`pandas_tools` is a Python package providing additional functionalities for pandas. In particular, it provides functionalities that enhance `DataFrame.to_latex` for generating publication-quality tables. 

## Installation

`pip install git+https://github.com/jiafengkevinchen/pandas_tools`

## Example

```python
import pandas_tools.latex as tex

# These SEs may be negative; for illustrative purposes only
df = pd.DataFrame(
    np.random.randn(4, 7),
    columns=["coef", "se1", "se2", "coef_", "se1_", "se2_", "other"],
    index=["arg1", "arg2", "arg3", "arg4"],
)
df.iloc[2, 2] = np.nan

print(
    df.consolidate_se(
        ["coef", "coef_"], ["se1", "se1_"], ["se2", "se2_"]
    ).to_latex_table(
        caption="this is a table",
        mathify_args=dict(texttt_column=True, texttt_index=True),
    )
)



```

outputs

```latex
\begin{table}[tbh]
        \caption{this is a table}
        \label{tab:this_is_a_table}
        \centering
        \vspace{1em}
        \begin{tabular}{lccc}
\toprule
{} &                              \texttt{coef} &                            \texttt{coef\_} &                             \texttt{other} \\
{} & \hypertarget{tabcol:this_is_a_table1}{(1)} & \hypertarget{tabcol:this_is_a_table2}{(2)} & \hypertarget{tabcol:this_is_a_table3}{(3)} \\
\midrule
\texttt{arg1} &                                   $-0.620$ &                                    $1.217$ &                                   $-1.211$ \\
              &                                 ($-0.328$) &                                 ($-0.643$) &                                            \\
              &                                  [$0.933$] &                                 [$-1.637$] &                                            \\
\texttt{arg2} &                                   $-2.708$ &                                   $-0.205$ &                                   $-0.014$ \\
              &                                 ($-0.574$) &                                  ($0.179$) &                                            \\
              &                                 [$-0.123$] &                                  [$0.572$] &                                            \\
\texttt{arg3} &                                   $-2.016$ &                                    $0.797$ &                                    $0.353$ \\
              &                                 ($-1.121$) &                                 ($-2.756$) &                                            \\
              &                                        --- &                                 [$-0.503$] &                                            \\
\texttt{arg4} &                                    $0.559$ &                                   $-0.177$ &                                    $1.187$ \\
              &                                  ($0.256$) &                                 ($-1.000$) &                                            \\
              &                                 [$-1.450$] &                                  [$0.936$] &                                            \\
\bottomrule
\end{tabular}

        \end{table}

```

