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
    ).to_latex_table(caption="my table")
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
{} &                                       coef &                                      coef_ &                                      other \\
{} & \hypertarget{tabcol:this_is_a_table1}{(1)} & \hypertarget{tabcol:this_is_a_table2}{(2)} & \hypertarget{tabcol:this_is_a_table3}{(3)} \\
\midrule
arg1 &                                    $0.463$ &                                    $0.527$ &                                   $-0.959$ \\
     &                                 ($-0.964$) &                                  ($0.084$) &                                            \\
     &                                 [$-1.125$] &                                 [$-1.291$] &                                            \\
arg2 &                                   $-0.252$ &                                   $-0.132$ &                                    $1.674$ \\
     &                                 ($-2.107$) &                                  ($0.012$) &                                            \\
     &                                  [$2.146$] &                                  [$1.397$] &                                            \\
arg3 &                                    $0.776$ &                                    $0.221$ &                                   $-1.079$ \\
     &                                  ($0.836$) &                                 ($-1.001$) &                                            \\
     &                                        --- &                                  [$0.859$] &                                            \\
arg4 &                                    $0.253$ &                                   $-0.915$ &                                   $-0.409$ \\
     &                                 ($-0.180$) &                                  ($0.940$) &                                            \\
     &                                 [$-1.483$] &                                 [$-0.276$] &                                            \\
\bottomrule
\end{tabular}

        \end{table}

```

