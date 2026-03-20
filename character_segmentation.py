import cv2
import numpy as np
from skimage.filters import threshold_sauvola


def get_perspective_transform(image, pts):
    """Straightens a tilted quadrilateral into a horizontal rectangle."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[1] - br[1]) ** 2) + ((tr[0] - br[0]) ** 2))
    heightB = np.sqrt(((tl[1] - bl[1]) ** 2) + ((tl[0] - bl[0]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped, maxHeight

def segment_characters(
    image_path,
    det_model,
    target_dim=1024,
    padding=2,
    sauvola_window=25,
    dot_height_ratio=0.25,
    max_dot_gap_ratio=0.8,
    horizontal_tolerance_ratio=0.3,
    min_area_threshold=5,
    dilation_iterations=1
):
    image = cv2.imread(image_path)
    # Resize
    h_orig, w_orig = image.shape[:2]
    scale = target_dim / max(h_orig, w_orig)
    image = cv2.resize(image, (int(w_orig * scale), int(h_orig * scale)))

    # Binarize Full Image (Sauvola)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh_val = threshold_sauvola(gray, window_size=sauvola_window)
    binary_inv = (gray < thresh_val).astype(np.uint8) * 255

    # Text Detection
    output = det_model.predict(image, batch_size=1)
    character_crops = []

    for res in output:
        polys = res['dt_polys']
        polys = sorted(polys, key=lambda p: np.mean(p[:, 1]))

        for poly in polys:
            # Rectify the tilted text line
            line_img, line_h = get_perspective_transform(image, poly.astype("float32"))
            line_bin, _ = get_perspective_transform(binary_inv, poly.astype("float32"))

            # Connect Broken Strokes
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            dilated = cv2.dilate(line_bin, dilate_kernel, iterations=dilation_iterations)
            
            # --- CONNECTED COMPONENTS ANALYSIS ---
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                dilated, connectivity=8
            )
            
            component_info = []
            # Skip label 0 because it is the background
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= min_area_threshold:
                    x = stats[i, cv2.CC_STAT_LEFT]
                    y = stats[i, cv2.CC_STAT_TOP]
                    w = stats[i, cv2.CC_STAT_WIDTH]
                    h = stats[i, cv2.CC_STAT_HEIGHT]
                    cx, cy = centroids[i]
                    
                    component_info.append({
                        'id': i,
                        'box': (x, y, w, h),
                        'center': (cx, cy),
                        'area': area,
                        'label_ids': [i],
                        'merged': False
                    })

            # Dynamic Dot Logic
            max_dot_dim = line_h * dot_height_ratio
            max_dot_area = max_dot_dim ** 2

            dots = [c for c in component_info if c['box'][3] < max_dot_dim and c['area'] < max_dot_area]
            bodies = [c for c in component_info if c not in dots]

            final_characters = []

            for dot in dots:
                if dot['merged']: continue
                dx, dy, dw, dh = dot['box']
                dcx, dcy = dot['center']
                
                best_body = None
                min_dist = float('inf')
                
                for body in bodies:
                    if body['merged']: continue
                    bx, by, bw, bh = body['box']
                    bcx, bcy = body['center']
                    
                    if by < (dy + dh): continue 
                    
                    tol = bw * horizontal_tolerance_ratio
                    if (bx - tol < dcx < bx + bw + tol):
                        v_gap = by - (dy + dh)
                        if 0 <= v_gap <= (bh * max_dot_gap_ratio):
                            dist = bcy - dcy
                            if dist < min_dist:
                                min_dist = dist
                                best_body = body
                
                if best_body:
                    bx, by, bw, bh = best_body['box']
                    nx, ny = min(dx, bx), min(dy, by)
                    nw = max(dx+dw, bx+bw) - nx
                    nh = max(dy+dh, by+bh) - ny
                    
                    dot['merged'] = True
                    best_body['merged'] = True
                    
                    # Store the merged box AND combine the component labels
                    final_characters.append({
                        'box': (nx, ny, nw, nh),
                        'label_ids': dot['label_ids'] + best_body['label_ids']
                    })

            # Add unmerged bodies
            for b in bodies:
                if not b['merged']: final_characters.append(b)
            
            # Sort Left-to-Right
            final_characters.sort(key=lambda char: char['box'][0])

            # Final Cropping with CCA ISOLATION MASKING
            for char in final_characters:
                cx, cy, cw, ch = char['box']
                
                char_mask = np.isin(labels, char['label_ids'])
                isolated_line = np.full_like(line_img, 255)
                isolated_line[char_mask] = line_img[char_mask]

                x1, y1 = max(0, cx - padding), max(0, cy - padding)
                x2, y2 = min(line_img.shape[1], cx + cw + padding), min(line_img.shape[0], cy + ch + padding)
                
                char_crop = isolated_line[y1:y2, x1:x2]
                if char_crop.size > 0:
                    character_crops.append(char_crop)

    return character_crops

def SC_main(image_file='examples/good_example.jpg', output_dir=''):
    # --- Parameters ---
    import os
    from paddleocr import TextDetection
    
    MODEL_NAME = "PP-OCRv5_mobile_det"
    MODEL_PATH = r"resources\PP-OCRv5_mobile_det_infer"
    det_model = TextDetection(model_name=MODEL_NAME, model_dir=MODEL_PATH, enable_mkldnn=False)
    
    list_of_char_images = segment_characters(
        image_file,
        det_model=det_model
    )

    if list_of_char_images:
        if output_dir:
            os.makedirs(output_dir, exist_ok=True) # Create directory if it doesn't exist
            print(f"Saving individual characters to '{output_dir}/'...")
            base_filename = os.path.splitext(os.path.basename(image_file))[0]
            for i, char_img in enumerate(list_of_char_images):
                save_path = os.path.join(output_dir, f"{base_filename}_char_{i}.png") # e.g., handwritten_alphabet_char_000.png
                cv2.imwrite(save_path, char_img)
            print("Saving complete.")

    else:
        print("\nNo characters were segmented from the image.")
    
    return list_of_char_images
    
if __name__ == "__main__":
    SC_main(image_file='examples/good_example.jpg', output_dir = "segmented_characters")