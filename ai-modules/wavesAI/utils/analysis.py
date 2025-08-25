import pandas as pd
import os
import wandb
from collections.abc import MutableMapping
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
import numpy as np

## Loads the correct tqdm depending on environment
try:
    shell = get_ipython().__class__.__name__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm


def flatten_dict(dictionary, parent_key='', separator='.'):
    """
    Flatten a nested dictionary

    :param dictionary: python dictionary to flatten
    :param parent_key:
    :param separator:
    :return: flattened dictionary
    """
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def get_runs_from_wandb(exp_name, wandb_project="wavesAI"):
    """
    Get runs from wandb. We have to use this instead of CSV export in WANDB as csv export is limited to 1000 rows

    :param exp_name: name of the experiment in wandb
    :param wandb_project: name of the wandb project (defaults to "wavesAI" - for this project)

    :return: pd.DataFrame of the experiment's runs
    """
    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs(f"{wandb_project}/{exp_name}")

    return_dicts = []
    summary_list, config_list, name_list = [], [], []
    for run in tqdm(runs):
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k, v in run.config.items()
             if not k.startswith('_')})

        # .name is the human-readable name of the run.
        name_list.append(run.name)

        # populating the run_dict with the wandb data
        run_dict = {'name': run.name}
        run_dict.update(flatten_dict(run.config))
        run_dict.update(flatten_dict(run.summary._json_dict))

        return_dicts.append(run_dict)

    runs_df = pd.DataFrame({
        "summary": summary_list,
        "config": config_list,
        "name": name_list
    })

    return pd.DataFrame(return_dicts)


def get_df_of_runs(exp_names):
    """
    Obtain a dataframe of runs from wandb or local filesystem when available

    :param exp_names: list of experiment names
    :return: pd.DataFrame of runs all concatenated together
    """
    df_list = []
    for exp_name in exp_names:
        print(f"Processing {exp_name}")
        filename = f"wandb_{exp_name}.csv"
        df_exp = None
        if os.path.exists(f"cache/{filename}"):
            print(f"loading from saved file for {exp_name}")
            df_exp = pd.read_csv(f"cache/{filename}")
        else:
            print(f"Requesting data from wandb for {exp_name}.")
            df_exp = get_runs_from_wandb(exp_name)
            df_exp.to_csv(f"cache/{filename}")

        df_list.append(df_exp)

    return pd.concat(df_list)


def plot_bar_comparison_chart_for_df(df, category_name, subcategory_name,
                                     report_column="test_accuracy", categories=None, subcategories=None,
                                     roundoff=3,
                                     ylabel=None, width=0.5, multiplier=0.0, legend_map=None):
    """
    Plot a 2-step comparison bar chart. The first level of comparison will be the different ticks in the x-axis
    and the second level will be the different bars within each x-axis tick

    CAUTION: NOT USED. BETTER PLOTS CAN BE OBTAINED WITH SEABORN CATPLOT

    :param df: pd.DataFrame to compare and plot.
    :param category_name: first level comparison column name in the DataFrame.
    :param subcategory_name: second level comparison column name in the DataFrame.
    :param report_column: The column name to report on the bars.
    :param categories: list or None. If None, all categories will be reported.
    :param subcategories: list or None. If None, all subcategories will be reported.
    :param roundoff: roundoff for the bars.
    :param ylabel: str or None. y-axis label. If None, the report_column.
    :param width: float. width of the bars.
    :param multiplier: float. spacing between categories.
    :param legend_map: Dict or None. If None, legend will be the same as the name of the subcategories.
    """
    ylabel = report_column if ylabel is None else ylabel
    # build categories and category dict
    categories = df[category_name].unique() if categories is None else categories
    subcategories = df[subcategory_name].unique() if subcategories is None else subcategories

    legend_map = { subcategory: subcategory for subcategory in subcategories } if legend_map is None else legend_map

    category_dict = {}
    category_dict_std = {}
    # for category in categories:
    for subcategory in subcategories:
        category_dict[subcategory] = []
        category_dict_std[subcategory] = []
        for category in categories:
            # print(df[(df[subcategory_name] == subcategory)].head())
            df_temp = df[df[subcategory_name] == subcategory]
            # print(df_temp.head())
            df_temp = df_temp[df_temp[category_name] == category]
            vals = df_temp[report_column]
            # print(category, subcategory, vals)
            # print(category, subcategory, vals)
            if len(vals) <= 0:
                print(f"Nothing found for category: {category} and subcategory: {subcategory}")
            val_mean = vals.mean()
            category_dict[subcategory].append(round(vals.mean(), roundoff))
            category_dict_std[subcategory].append([round(val_mean-vals.min(), roundoff), round(vals.max()-val_mean, roundoff)])

    # print(category_dict)

    x = np.arange(len(categories))*multiplier  # the label locations

    # fig, ax = plt.subplots(layout='constrained')
    fig = plt.gcf()
    ax = plt.gca()
    step = 0
    for attribute, measurements in category_dict.items():
        # print(att)
        # print(attribute, measurement)
        minmax = np.array(category_dict_std[attribute])
        print(minmax.T)
        offset = width * step
        rects = ax.bar(x + offset, measurements, width, label=legend_map[attribute],
                       yerr=minmax.T)
        ax.bar_label(rects, padding=3)
        step += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel)
    # ax.set_title('Penguin attributes by categories')
    ax.set_xticks(x + width, categories)
    ax.legend(loc='upper left', ncols=3)
    # ax.set_ylim(0.5, 1.1)

    # plt.show()


def add_values_to_sns_bars(g):
    # iterate through axes
    for ax in g.axes.ravel():
        # add annotations
        for c in ax.containers:
            # add custom labels with the labels=labels parameter if needed
            # labels = [f'{h}' if (h := v.get_height()) > 0 else '' for v in c]
            # print(c.datavalues)
            c.datavalues = [round(val, 3) for val in c.datavalues]
            print(c.datavalues)
            ax.bar_label(c, label_type='edge', fmt="{:.2f}", padding=40)
        ax.margins(y=0.2)
