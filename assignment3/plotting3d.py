import matplotlib.pyplot as plt
import pathlib
import json
import utils


data = {}
with open("models.json", "r") as file:
    data = json.load(file)
with_batch = data["with_batch"]["train"]["history"]["loss"]
wo_batch = data["wo_batch"]["train"]["history"]["loss"]

with_batch_val = data["with_batch"]["validation"]["history"]["loss"]
wo_batch_val = data["wo_batch"]["validation"]["history"]["loss"]
plt.figure(figsize=(20, 8))
plt.subplot(1,2,1)
plt.title("Train loss comparison")
utils.plot_loss(with_batch, label="With batch normalization", npoints_to_average=10)
utils.plot_loss(wo_batch, label="Without batch normalization", npoints_to_average=10)
plt.legend()
plt.subplot(1,2,2)
plt.title("Validation loss comparison")
utils.plot_loss(with_batch_val, label="With batch normalization")
utils.plot_loss(wo_batch_val, label="Without batch normalization")
plt.legend()
plot_path = pathlib.Path("plots")
plot_path.mkdir(exist_ok=True)
plt.savefig(plot_path.joinpath(f"feature_comparison_plot.png"))
plt.show()
