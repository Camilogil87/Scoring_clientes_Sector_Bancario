import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def transform_rgba_color_scale_from_255_to_1(color_scale):
    transformed_scale = list()
    for color in color_scale:
        transformed_color = list()
        for i, element in enumerate(color):
            if i < 3:
                transformed_color.append(element/255)
            else:
                transformed_color.append(element)
        transformed_scale.append(transformed_color)
    return transformed_scale


def plot_color_scales(title, cmap_list):
    nrows = len(cmap_list)
    fig, axes = plt.subplots(nrows=nrows, figsize=(6.4, 0.26 * nrows))
    fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
    axes[0].set_title(title, y=1.05)

    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    for ax, name in zip(axes, cmap_list):
        ax.imshow(gradient, aspect='auto', cmap=name)
        pos = list(ax.get_position().bounds)
        x_text = pos[0] - 0.01
        y_text = pos[1] + pos[3] / 2.
        fig.text(x_text, y_text, name, va='center', ha='right')

    for ax in axes:
        ax.set_axis_off()
