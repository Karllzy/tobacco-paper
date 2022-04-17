from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from scipy.signal import savgol_filter
import pickle

extract_sample_number = 20

current_window, current_order = 15, 0


def main():
    with open("./dataset/draw_pic.p", 'rb') as f:
        d = pickle.load(f)
    select_tobacco_sample, selected_bands = d
    filtered_tobacco_sample = savgol_filter(select_tobacco_sample, window_length=current_window, polyorder=current_order)
    fig, axs = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(8, 4)
    axs[0].plot(selected_bands, select_tobacco_sample.T)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].set_xlabel("wavelength(nm)")
    axs[0].set_ylabel("signal density")
    axs[0].set_title(f"{extract_sample_number} tobacco samples before filtering")
    p = axs[1].plot(selected_bands, filtered_tobacco_sample.T)
    axs[1].set_xlabel("wavelength(nm)")
    axs[1].set_ylabel("signal density")
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].set_title(f"{extract_sample_number} tobacco samples after filtering")
    plt.subplots_adjust(bottom=0.25)
    ax_slide_win = plt.axes([0.2, 0.05, 0.65, 0.03])  # xposition, yposition, width and height
    ax_slide_order = plt.axes([0.2, 0.1, 0.65, 0.03])  # xposition, yposition, width and height
    # Properties of the slider
    win_size = Slider(ax_slide_win, 'Window size', valmin=3, valmax=60, valinit=15, valstep=2)
    order = Slider(ax_slide_order, 'Order', valmin=0, valmax=9, valinit=0, valstep=1)
    # Updating the plot


    def update_win(_):
        global current_window
        current_window = int(win_size.val)
        fresh()


    def update_order(_):
        global current_order
        current_order = int(order.val)
        fresh()


    def fresh():
        global current_order, current_window
        new_y = savgol_filter(select_tobacco_sample, current_window, current_order)
        for idx, line in enumerate(p):
            line.set_ydata(new_y[idx, :].T)
        fig.canvas.draw()  # redraw the figure


    win_size.on_changed(update_win)
    order.on_changed(update_order)
    plt.show()


if __name__ == '__main__':
    main()