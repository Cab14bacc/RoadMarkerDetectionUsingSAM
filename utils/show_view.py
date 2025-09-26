import numpy as np
import matplotlib.pyplot as plt
import cv2

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_mask_with_point_cv2(image, masks, scores, input_point, input_label, no_show=False):
    """Fast version show_mask_with_point using only OpenCV"""
    ret_images = []
    
    # Resize only once
    max_dim = 800
    h, w = image.shape[:2]
    scale = min(max_dim/h, max_dim/w)
    new_size = (int(w*scale), int(h*scale))
    resized_image = cv2.resize(image, new_size)
    
    
    for i, (mask, score) in enumerate(zip(masks, scores)):
        # Create a copy for this mask
        vis_image = resized_image.copy()
        
        # Resize mask to match the image
        resized_mask = cv2.resize(mask.astype(np.uint8), new_size)
        
        # Create semi-transparent overlay for mask
        mask_overlay = np.zeros_like(vis_image)
        mask_overlay[resized_mask > 0] =  [30, 144, 255]
        
        # Apply mask with transparency
        alpha = 0.5
        beta = 0.8
        vis_image = cv2.addWeighted(mask_overlay, alpha, vis_image, beta, 0)
        
        # Draw points
        pos_points = input_point[input_label==1]
        neg_points = input_point[input_label==0]
        
        # Scale points to match resized image
        pos_points = pos_points * scale
        neg_points = neg_points * scale
        
        # Draw positive points (green stars)
        for point in pos_points:
            x, y = int(point[0]), int(point[1])
            cv2.drawMarker(vis_image, (x, y), (0, 255, 0), cv2.MARKER_STAR, 20, 2)
            
        # Draw negative points (red stars)
        for point in neg_points:
            x, y = int(point[0]), int(point[1])
            cv2.drawMarker(vis_image, (x, y), (255, 0, 0), cv2.MARKER_STAR, 20, 2)
        
        # Add text for score
        cv2.putText(vis_image, f"Mask {i+1}, Score: {score:.3f}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if no_show:
            ret_images.append(vis_image)
        else:
            cv2.imshow(f"Mask {i+1}", vis_image)
            cv2.waitKey(0)
            cv2.destroyWindow(f"Mask {i+1}")
            
    if no_show:
        return np.array(ret_images)
    
def show_mask_with_point(image, masks, scores, input_point, input_label, no_show=False):
    if no_show:
        ret_images = []

    resized_image = cv2.resize(image, (800, 800))

    for i, (mask, score) in enumerate(zip(masks, scores)):
        fig = plt.figure(figsize=(8,8), dpi=100)
        plt.imshow(resized_image)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(label=f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        if no_show:
            fig.canvas.draw()
            fig_width, fig_height = fig.get_size_inches() * fig.get_dpi()
            fig_width, fig_height = int(fig_width), int(fig_height)
            ret_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            ret_image = ret_image.reshape(fig_height, fig_width, 3)
            ret_images.append(ret_image)
        else:
            plt.show()

        plt.close('all')

    if no_show:
        return np.array(ret_images)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    