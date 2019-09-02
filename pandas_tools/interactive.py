import ipywidgets as widgets
from IPython.display import display
from IPython.html.widgets import interactive


def display_columns(df):
    """
    Display the columns of df in a SelectMultiple widget and print the code
    representing the list of selected columns.
    """
    dt = list(df.dtypes.items())
    column_display = [f"{c} ({str(t)})" for c, t in dt]
    column = [c for c, t in dt]
    w = widgets.SelectMultiple(options=list(zip(column_display, column)))
    h = widgets.BoundedIntText(value=5, min=1, max=len(df))

    def print_columns(columns=None, head=5):
        if columns is None:
            return
        else:
            inner = map(lambda t: '"' + str(t) + '"', columns)
            print(f"cols_selected = [{', '.join(inner)}]")
            print()
            if len(columns) == 0:
                return
            try:
                display(df[list(columns)].head(head).style)
            except AttributeError:
                pass

    return interactive(print_columns, columns=w, head=h)


def rename_dict(df):
    """
    Provide a GUI for genearting the dict to pass into df.rename(columns=r_dict)
    """

    def tgg(b):
        if b.alt_text.value != b.text.value:
            b.alt_text.value = b.text.value
        else:
            b.alt_text.value = b.text.value.lower().strip().replace(" ", "_")

    lst = []
    button = widgets.Button(description="Add column", icon="plus")
    finish = widgets.Button(description="Get recipe", icon="check")
    out = widgets.Output()
    out2 = widgets.Output()

    def on_button_clicked(b):
        with out:
            ind = min(len(lst), len(df.columns) - 1)
            text = widgets.Dropdown(
                index=ind, description="Old name", options=df.columns
            )
            alt_text = widgets.Text(
                description="New name",
                value=df.columns[ind].lower().strip().replace(" ", "_"),
            )
            tg = widgets.Button(description="Toggle suggestion")
            tg.alt_text = alt_text
            tg.text = text
            tg.on_click(tgg)

            lst.append((text, alt_text))
            display(widgets.HBox([text, alt_text, tg]))
        out2.clear_output()

    def display_dict(b):
        out2.clear_output()
        with out2:
            s1 = repr({k.value: v.value for k, v in lst})
            print(f"df.rename(columns={s1})")

    button.on_click(on_button_clicked)
    finish.on_click(display_dict)

    w = widgets.VBox([button, out, finish, out2])
    display(w)
