# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
# from segment_anything import sam_model_registry, SamPredictor
 
# def show_mask(mask, ax, random_color=False):
#     if random_color:
#         color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
#     else:
#         color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
#     h, w = mask.shape[-2:]
#     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#     ax.imshow(mask_image)
 
 
# def show_points(coords, labels, ax, marker_size=375):
#     pos_points = coords[labels == 1]
#     neg_points = coords[labels == 0]
#     ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
#                linewidth=1.25)
#     ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
#                linewidth=1.25)
    

# sam_checkpoint = "/home/ubuntu/foundation_pose/FoundationPose-main/SAM/sam_vit_h_4b8939.pth"
# device = "cuda"
# model_type = "default"

# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam.to(device=device)
# predictor = SamPredictor(sam)

# image = cv2.imread(r"/home/ubuntu/foundation_pose/FoundationPose-main/demo_data/cub/rgb/1741937205907.png")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
# predictor.set_image(image)
 
# input_point = np.array([[1600, 1000]])
# input_label = np.array([1])
 
# plt.figure(figsize=(10,10))
# plt.imshow(image)
# show_points(input_point, input_label, plt.gca())
# plt.axis('on')
# plt.show()

# masks, scores, logits = predictor.predict(
#     point_coords=input_point,
#     point_labels=input_label,
#     multimask_output=True,
# )
 
# # 遍历读取每个扣出的结果
# for i, (mask, score) in enumerate(zip(masks, scores)):
#     plt.figure(figsize=(10,10))
#     plt.imshow(image)
#     show_mask(mask, plt.gca())
#     show_points(input_point, input_label, plt.gca())
#     plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
#     plt.axis('off')
#     plt.show()

# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
# from segment_anything import sam_model_registry, SamPredictor

# def show_mask(mask, ax, random_color=False):
#     if random_color:
#         color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
#     else:
#         color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
#     h, w = mask.shape[-2:]
#     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#     ax.imshow(mask_image)


# def show_points(coords, labels, ax, marker_size=375):
#     pos_points = coords[labels == 1]
#     neg_points = coords[labels == 0]
#     ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
#                linewidth=1.25)
#     ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
#                linewidth=1.25)

# sam_checkpoint = "/home/ubuntu/foundation_pose/FoundationPose-main/SAM/sam_vit_h_4b8939.pth"
# device = "cuda"
# model_type = "default"

# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam.to(device=device)
# predictor = SamPredictor(sam)

# image = cv2.imread(r"/home/ubuntu/foundation_pose/FoundationPose-main/demo_data/cub/rgb/1741937174624.png")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# predictor.set_image(image)

# # 交互式选择点
# fig, ax = plt.subplots(figsize=(10, 10))
# ax.imshow(image)
# plt.axis('on')

# input_points = []
# input_labels = []

# def onclick(event):
#     if event.inaxes == ax:
#         x, y = int(event.xdata), int(event.ydata)
#         input_points.append([x, y])
#         input_labels.append(1)
#         ax.scatter(x, y, color='green', marker='*', s=375, edgecolor='white', linewidth=1.25)
#         fig.canvas.draw()

# cid = fig.canvas.mpl_connect('button_press_event', onclick)
# plt.show()

# input_point = np.array(input_points)
# input_label = np.array(input_labels)

# masks, scores, logits = predictor.predict(
#     point_coords=input_point,
#     point_labels=input_label,
#     multimask_output=True,
# )

# # 遍历读取每个扣出的结果
# for i, (mask, score) in enumerate(zip(masks, scores)):
#     plt.figure(figsize=(10, 10))
#     plt.imshow(image)
#     show_mask(mask, plt.gca())
#     show_points(input_point, input_label, plt.gca())
#     plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
#     plt.axis('off')
#     plt.show()

import cv2
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import sam_model_registry, SamPredictor


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


sam_checkpoint = "/home/ubuntu/foundation_pose/FoundationPose-main/SAM/sam_vit_h_4b8939.pth"
device = "cuda"
model_type = "default"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

image = cv2.imread(r"/home/ubuntu/foundation_pose/FoundationPose-main/demo_data/kettle/rgb/1741938458747.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictor.set_image(image)

# 交互式选择点
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(image)
plt.axis('on')

input_points = []
input_labels = []


def onclick(event):
    if event.inaxes == ax:
        x, y = int(event.xdata), int(event.ydata)
        input_points.append([x, y])
        input_labels.append(1)
        ax.scatter(x, y, color='green', marker='*', s=375, edgecolor='white', linewidth=1.25)
        fig.canvas.draw()


cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

input_point = np.array(input_points)
input_label = np.array(input_labels)

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

# 遍历读取每个扣出的结果
for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()

    # 保存 mask 为图像
    mask_filename = f"mask_{i + 1}_score_{score:.3f}.png"
    binary_mask = (mask * 255).astype(np.uint8)
    cv2.imwrite(mask_filename, binary_mask)
    print(f"Mask {i + 1} saved as {mask_filename}")
