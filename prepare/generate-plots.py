import pandas as pd
import os

root_dir = os.path.abspath("")
dir_data = os.path.join(root_dir, "data")
dir_plots = os.path.join(root_dir, "plots")

# for filename in os.listdir(dir_data):
#    if filename.endswith('.log'):
#        base = os.path.splitext(filename)[0]
#        os.rename(dir_data + "/" + filename, dir_data + "/" + base + ".csv")


for filename in os.listdir(dir_data):
    if filename.endswith('.csv'):
        train = os.path.join(dir_data, filename)
        print(train)
        data = pd.read_csv(train, names=["time", "xAxis", "yAxis", "zAxis"])
        # data.head(10)
        # plot = data.plot("time", legend=None)
        plot = data.plot("time")
        # plot = data.plot("time", legend=None, cmap='gray')
        # [plot.lines[i].set_color(color) for i, color in enumerate(['red', 'blue', 'green'])]
        # plot.axis('off')
        fig = plot.get_figure()
        # fig.set_size_inches(5, 5)
        # fig.savefig(dir_plots + "/" + os.path.splitext(filename)[0] + ".png", transparent=True)
        fig.savefig(dir_plots + "/" + os.path.splitext(filename)[0] + ".png")
