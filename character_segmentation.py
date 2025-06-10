import cv2
import os

def segment_characters(
    image_path,
    target_dim=1024,
    padding=1,
    block_size=15,
    C=5,
    use_morph_open=True,
    open_kernel_size=(2, 2),
    dilation_iterations=1,
    dilation_kernel_size=(2, 2),
    min_area_threshold=5,
    dot_max_area=60,
    max_dot_gap_ratio=0.8,
    horizontal_tolerance_ratio=0.3
    ):
    """
    Segments handwritten characters and returns a list of cropped character images.

    Steps:
    1. Resizes image.
    2. Applies adaptive thresholding, morphological operations.
    3. Merges 'i'/'j' dots.
    4. Calculates final bounding boxes with padding.
    5. Crops characters from the resized image based on final boxes.

    Returns:
        list: A list of NumPy arrays, each representing a cropped character image.
              Returns an empty list if the image cannot be loaded or no characters are found.
    """

    image = cv2.imread(image_path)

    h_orig, w_orig = image.shape[:2]
    if h_orig >= w_orig:
        scale = target_dim / h_orig
        w_new = int(round(w_orig * scale))
        h_new = target_dim
    else:
        scale = target_dim / w_orig
        h_new = int(round(h_orig * scale))
        w_new = target_dim

    image = cv2.resize(image, (w_new, h_new), interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, C
    )

    if use_morph_open:
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, open_kernel_size)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_kernel)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, dilation_kernel_size)
    dilated = cv2.dilate(thresh, dilate_kernel, iterations=dilation_iterations)

    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_info = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area >= min_area_threshold:
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2
            contour_info.append({'id': i, 'contour': cnt, 'area': area, 'box': (x, y, w, h), 'center': (cx, cy), 'merged': False})

    potential_dots = [c for c in contour_info if c['area'] <= dot_max_area]
    potential_bodies = [c for c in contour_info if c['area'] > dot_max_area]
    merged_boxes = []

    for dot in potential_dots:
        if dot['merged']: continue
        dot_x, dot_y, dot_w, dot_h = dot['box']
        dot_cx, dot_cy = dot['center']
        dot_bottom = dot_y + dot_h
        best_match_body = None
        min_dist = float('inf')
        for body in potential_bodies:
            if body['merged']: continue
            body_x, body_y, body_w, body_h = body['box']
            body_cx, body_cy = body['center']
            if body_y < dot_bottom: continue
            tolerance = body_w * horizontal_tolerance_ratio
            if not (body_x - tolerance < dot_cx < body_x + body_w + tolerance): continue
            vertical_gap = body_y - dot_bottom
            max_gap = body_h * max_dot_gap_ratio
            if vertical_gap < 0 or vertical_gap > max_gap: continue
            dist = body_cy - dot_cy
            if dist < min_dist:
                min_dist = dist
                best_match_body = body
        if best_match_body:
            body_x, body_y, body_w, body_h = best_match_body['box']
            new_x = min(dot_x, body_x)
            new_y = min(dot_y, body_y)
            new_w = max(dot_x + dot_w, body_x + body_w) - new_x
            new_h = max(dot_y + dot_h, body_y + body_h) - new_y
            merged_boxes.append((new_x, new_y, new_w, new_h))
            dot['merged'] = True
            best_match_body['merged'] = True

    final_bounding_boxes_raw = []
    for box in merged_boxes:
        final_bounding_boxes_raw.append(box)
    for body in potential_bodies:
        if not body['merged']:
            final_bounding_boxes_raw.append(body['box'])

    if not final_bounding_boxes_raw:
         return []

    final_bounding_boxes_raw.sort(key=lambda box: (box[1], box[0]))

    character_crops = []

    for i, (x, y, w, h) in enumerate(final_bounding_boxes_raw):
        padded_x1 = x - padding
        padded_y1 = y - padding
        padded_x2 = x + w - padding
        padded_y2 = y + h - padding

        final_x = max(0, padded_x1)
        final_y = max(0, padded_y1)
        final_x2_clamped = min(w_new, padded_x2)
        final_y2_clamped = min(h_new, padded_y2)
        final_w = final_x2_clamped - final_x
        final_h = final_y2_clamped - final_y

        if final_w > 0 and final_h > 0:
            crop = image[final_y : final_y + final_h, final_x : final_x + final_w]
            character_crops.append(crop)

    return character_crops

def SC_main(image_file='images/test1.jpg'):
    # --- Parameters ---
    target_dimension = 1024
    pad_amount = 1
    block = 15
    const_C = 5
    apply_opening = True
    opening_k_size = (2, 2)
    dilate_iter = 1
    dilate_k_size = (2, 2)
    min_area = 5
    dot_area = 60
    dot_gap_ratio = 0.8
    h_tolerance_ratio = 0.3

    list_of_char_images = segment_characters(
        image_file,
        target_dim=target_dimension,
        padding=pad_amount,
        block_size=block,
        C=const_C,
        use_morph_open=apply_opening,
        open_kernel_size=opening_k_size,
        dilation_iterations=dilate_iter,
        dilation_kernel_size=dilate_k_size,
        min_area_threshold=min_area,
        dot_max_area=dot_area,
        max_dot_gap_ratio=dot_gap_ratio,
        horizontal_tolerance_ratio=h_tolerance_ratio
    )

    if list_of_char_images:

        output_dir = "segmented_characters"
        os.makedirs(output_dir, exist_ok=True) # Create directory if it doesn't exist
        print(f"Saving individual characters to '{output_dir}/'...")
        base_filename = os.path.splitext(os.path.basename(image_file))[0]
        for i, char_img in enumerate(list_of_char_images):
            save_path = os.path.join(output_dir, f"{base_filename}_char_{i}.png") # e.g., handwritten_alphabet_char_000.png
            cv2.imwrite(save_path, char_img)
        print("Saving complete.")

    else:
        print("\nNo characters were segmented from the image.")
    
if __name__ == "__main__":
    SC_main(image_file='images/test1.jpg') # Change to your image path