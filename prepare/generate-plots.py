import pandas as pd
import os

root_dir = os.path.abspath("")
dir_data = os.path.join(root_dir, "data")
dir_plots = os.path.join(root_dir, "plots")

for filename in os.listdir(dir_data):
    if filename.endswith('.csv'):
        train = os.path.join(dir_data, filename)
        print(train)
        data = pd.read_csv(train, names=["time", "xAxis", "yAxis", "zAxis"])
        plot = data.plot("time")
        fig = plot.get_figure()
        fig.savefig(dir_plots + "/" + os.path.splitext(filename)[0] + ".png")
