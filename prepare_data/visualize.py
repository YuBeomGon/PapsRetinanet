import matplotlib.pyplot as plt
import cv2

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White

def visualize_bbox(img, bbox, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
#     x_min, y_min, x_max, y_max = list(map(int, bbox))
#     print(bbox)
    if len(bbox) == 4:
        x_min, y_min, w, h = (bbox)
    else :
        x_min, y_min, w, h, label = (bbox)
    x_max = x_min + w
    y_max = y_min + h
#     x_min, y_min, x_max, y_max = list(map(round, bbox))

    img = cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=BOX_COLOR, thickness=thickness)
    return img

def visualize(image, bboxes):
    img = image.copy()
#     img = image.clone().detach()
    for bbox in (bboxes):
#         print(bbox)
        img = visualize_bbox(img, bbox)
    plt.figure(figsize=(7, 7))
    plt.axis('off')
    plt.imshow(img)
    
    
def plot_images(img_list) :
    # img_list = []

    fig = plt.figure(figsize=(16, 16))
    fig.tight_layout()

    rows = 2
    cols = 2

    ax2 = fig.add_subplot(rows, cols, 0)
    ax2.imshow(img_list[0])
    ax2.set_title('img2')
    ax2.axis("off")

    ax3 = fig.add_subplot(rows, cols, 1)
    ax3.imshow(img_list[1])
    ax3.set_title('img3')
    ax3.axis("off")

    ax4 = fig.add_subplot(rows, cols, 2)
    ax4.imshow(img_list[2])
    ax4.set_title('img4')
    ax4.axis("off")

    ax5 = fig.add_subplot(rows, cols, 3)
    ax5.imshow(img_list[3])
    ax5.set_title('img5')
    ax5.axis("off")

    plt.show()    
        