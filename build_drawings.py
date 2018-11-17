import pandas as pd
import numpy as np
import ast
import draw_lines

def get_sample_imgs(num_img_to_show, data):
    num_imgs = data.shape[0]
    data = data.iloc[np.random.choice(num_imgs, size=num_img_to_show, replace=False), :]
    data['drawing'] = data['drawing'].apply(ast.literal_eval)
    return data['drawing']

def build_drawing(drawing,size):
    img = np.zeros((size,size))
    for stroke in drawing:
        for i in range(len(stroke[0]) - 1):
            x_coords, y_coords = draw_lines.get_line_int_coords(stroke[0][i], stroke[1][i], stroke[0][i + 1], stroke[1][i + 1])
            img[x_coords, y_coords] = 1
    return img


def plot_sample_imgs(num_img_to_show,data):
    import matplotlib.pyplot as plt
    n = int(np.sqrt(num_img_to_show))
    fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=(12, 12))
    for i in range(num_img_to_show):
        ax = axs[i // n, i % n]
        ax.imshow(build_drawing(data.iloc[i],256), cmap=plt.cm.gray)
        ax.axis('off')
        # plt.imshow(build_drawing(data.iloc[i],256), cmap=plt.cm.gray)
        # plt.axis('off')
        # plt.show()
    plt.tight_layout()
    fig.savefig('samples.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    num_img_to_show = 64
    data = pd.read_csv('train_simplified/zebra.csv')
    data = get_sample_imgs(num_img_to_show, data)
    plot_sample_imgs(num_img_to_show,data)