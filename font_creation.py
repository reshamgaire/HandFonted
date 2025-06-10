import numpy as np
from fontTools.ttLib import TTFont, newTable
from fontTools.pens.ttGlyphPen import TTGlyphPen
from skimage.measure import find_contours
from skimage.morphology import skeletonize, remove_small_objects
import os
from typing import List, Tuple
import cv2

def estimate_side_bearing_weights(binary_image):
    """
    Estimates LSB and RSB weights for a character in a binary image using skimage.

    Args:
        binary_image: A 2D NumPy array (binary, can be 0/1 or 0/255).

    Returns:
        (lsb_weight, rsb_weight): Tuple of estimated LSB and RSB weights.
    """

    rows, cols = np.where(binary_image > 0)

    if rows.size == 0:
        return 0, 0

    by = np.min(rows)
    bx = np.min(cols)
    bh = np.max(rows) - by + 1
    bw = np.max(cols) - bx + 1

    if bw == 0 or bh == 0:
        return 0.5, 0.5

    left_indents = []
    right_indents = []

    char_pixels_only_roi = binary_image[by:by+bh, bx:bx+bw]

    for r in range(bh):
        row_pixels = char_pixels_only_roi[r, :]
        nz_indices = np.nonzero(row_pixels)[0]

        if len(nz_indices) > 0:
            first_char_x_in_row_relative_to_roi = nz_indices[0]
            last_char_x_in_row_relative_to_roi = nz_indices[-1]

            left_indents.append(first_char_x_in_row_relative_to_roi)
            right_indents.append((bw - 1) - last_char_x_in_row_relative_to_roi)

    avg_left_indent = np.mean(left_indents) if left_indents else 0
    avg_right_indent = np.mean(right_indents) if right_indents else 0

    weight_for_lsb = max(1.0, (bw / 2.0) - avg_left_indent)
    weight_for_rsb = max(1.0, (bw / 2.0) - avg_right_indent)

    total_weight = weight_for_lsb + weight_for_rsb
    if total_weight < 2.0:
        weight_for_lsb = 1.0
        weight_for_rsb = 1.0
        total_weight = 2.0

    return weight_for_lsb / total_weight, weight_for_rsb / total_weight

def make_uniform_thickness(image, output_height=100, target_thickness=7): 
    """
    Convert a binary character image to uniform thickness font-like character using Distance Transform.
    
    Args:
        image (numpy.ndarray): Input binary character image.
        output_height (int): Target height for the character canvas before padding.
        target_thickness (int): Desired stroke thickness.
    
    Returns:
        numpy.ndarray: Processed character image.
    """
    
    output_width = output_height*image.shape[1]//image.shape[0]

    img_resized = cv2.resize(image, (output_width, output_height), interpolation=cv2.INTER_LANCZOS4)
    _, img_resized_binary = cv2.threshold(img_resized, 127, 255, cv2.THRESH_BINARY)

    skeleton_input_bool = img_resized_binary > 127
    skeleton_bool = skeletonize(skeleton_input_bool)
    
    skeleton_uint8 = (skeleton_bool.astype(np.uint8) * 255)

    uniform_char_uint8 = np.zeros_like(skeleton_uint8)

    if np.count_nonzero(skeleton_uint8) == 0:
        uniform_char_uint8 = skeleton_uint8.copy()
    else:
        radius = target_thickness / 2.0 
        img_inv_for_dt = (skeleton_bool == False).astype(np.uint8)
        dt = cv2.distanceTransform(img_inv_for_dt, cv2.DIST_L2, cv2.DIST_MASK_5)
        uniform_char_bool = dt <= radius 
        uniform_char_uint8 = (uniform_char_bool.astype(np.uint8) * 255)

    uniform_char_bool_cleaned = uniform_char_uint8 > 127

    min_artifact_size = max(10, int(target_thickness * 1.5))
    uniform_char_bool_cleaned = remove_small_objects(uniform_char_bool_cleaned, min_size=min_artifact_size)
    
    result = (uniform_char_bool_cleaned.astype(np.uint8) * 255)

    extra_padding_amount = 1
    result = cv2.copyMakeBorder(result, extra_padding_amount, extra_padding_amount,
                                    extra_padding_amount, extra_padding_amount,
                                    cv2.BORDER_CONSTANT, value=0)
    
    return result

def set_font_names(font: TTFont, family_name: str, style_name: str = "Regular"):
    """
    Sets essential font names in the 'name' table using a simplified approach.

    Args:
        font: The TTFont object.
        family_name: The desired font family name (e.g., "My Custom Font").
        style_name: The style (e.g., "Regular", "Bold").
    """
    name_table = font['name']

    full_font_name = f"{family_name} {style_name}".strip()
    ps_family_name = "".join(c for c in family_name if c.isalnum()) or "UnknownFamily"
    ps_style_name = "".join(c for c in style_name if c.isalnum()) or "Regular"
    ps_name = f"{ps_family_name}-{ps_style_name}"
    

    names_to_set = {
        1: family_name,       # Font Family
        2: style_name,        # Font Subfamily
        4: full_font_name,    # Full Font Name
        6: ps_name,           # PostScript Name
    }

    for nameID, name_string in names_to_set.items():
        # For Windows:
        name_table.setName(name_string, nameID, platformID=3, platEncID=1, langID=0x0409)
        # For Mac:
        name_table.setName(name_string, nameID, platformID=1, platEncID=0, langID=0)


def update_font_from_images(
    font_path: str,
    char_image_list: List[Tuple[str, np.ndarray]],
    output_path: str,
    desired_thickness: int = 100,
    new_family_name: str = None,
    new_style_name: str = "Regular"
):
    """
    Updates multiple glyphs in a TrueType font using corresponding Image.
    Scales and positions new glyphs to match original glyphs' yMin, yMax, LSB, and Advance Width.

    Args:
        font_path: Path to the input .ttf font file.
        char_image_list: A list of tuples, where each tuple is (character_string, numpy.ndarray).
        output_path: Path to save the modified .ttf font file.
    """
    try:
        font = TTFont(font_path)

        cmap = font.getBestCmap()
        if not cmap:
            return

        glyf_table = font['glyf']
        hmtx_table = font['hmtx']
        tables_to_remove = ['prep', 'cvt ', 'hdmx', 'LTSH', 'VDMX']
        for table_tag in tables_to_remove:
            if table_tag in font:
                del font[table_tag]

        resized_height = 100

        for char_val, image in char_image_list:
            char_code = ord(char_val)
            if char_code not in cmap:
                continue
            glyph_name = cmap[char_code]

            if glyph_name not in glyf_table:
                continue

            original_advance_width, original_lsb = None, None
            try:
                original_advance_width, original_lsb = hmtx_table[glyph_name]
            except KeyError:
                continue
            if original_advance_width is None or original_lsb is None:
                continue

            original_glyph = glyf_table[glyph_name]
            original_glyph.recalcBounds(glyf_table)
            original_ymin = original_glyph.yMin
            original_ymax = original_glyph.yMax
            original_xmin = original_glyph.xMin
            original_xmax = original_glyph.xMax

            original_height = original_ymax - original_ymin
            original_glyph_width = original_xmax - original_xmin
            target_ymin = original_ymin


            height_factor = original_height/resized_height
            resized_thickness = int(desired_thickness/height_factor)
            binary_image = make_uniform_thickness(image,output_height=resized_height, target_thickness=resized_thickness)

            contours = find_contours(binary_image, level=0.5)
            if not contours:
                continue

            all_points = np.concatenate(contours, axis=0)
            min_img_y_px, min_img_x_px = np.min(all_points, axis=0)
            max_img_y_px, max_img_x_px = np.max(all_points, axis=0)
            img_height_px = max_img_y_px - min_img_y_px
            if img_height_px <= 0:
                continue

            scale_factor = original_height / img_height_px
            
            final_contours_for_glyph = []
            for contour_pts_raw in contours:
                img_y_coords = contour_pts_raw[:, 0]
                img_x_coords = contour_pts_raw[:, 1]
                
                glyph_points_x = (img_x_coords - min_img_x_px) * scale_factor
                scaled_flipped_y_relative_to_img_top = (max_img_y_px - img_y_coords) * scale_factor
                glyph_points_y = scaled_flipped_y_relative_to_img_top + target_ymin
                
                final_contour_pts = np.vstack((glyph_points_x, glyph_points_y)).T
                final_contours_for_glyph.append(final_contour_pts)
            
            pen = TTGlyphPen(glyphSet=None)
            for contour_pts in final_contours_for_glyph:
                if not np.allclose(contour_pts[0], contour_pts[-1], atol=1e-2):
                     contour_pts = np.vstack((contour_pts, contour_pts[0]))
                pen.moveTo(tuple(contour_pts[0]))
                for pt in contour_pts[1:]: pen.lineTo(tuple(pt))
                pen.closePath()
            new_glyph = pen.glyph()

            glyf_table[glyph_name] = new_glyph
            new_glyph.recalcBounds(glyf_table)

            new_glyph_ink_width = new_glyph.xMax - new_glyph.xMin
            if new_glyph_ink_width < 0: new_glyph_ink_width = 0

            lsb_weight, rsb_weight = estimate_side_bearing_weights(image)
            
            if new_glyph_ink_width > original_glyph_width:
                total_bearing = max(10, original_advance_width - new_glyph_ink_width)
            else:
                total_bearing = max(10, original_advance_width - original_glyph_width)
            
            total_bearing = min(total_bearing, new_glyph_ink_width*0.2)
            final_lsb = int(total_bearing * lsb_weight)
            final_rsb = int(total_bearing * rsb_weight)
            final_advance_width = new_glyph_ink_width + final_lsb + final_rsb
            
            hmtx_table[glyph_name] = (final_advance_width, final_lsb)
        
        if new_family_name:
            if 'name' not in font:
                font['name'] = newTable('name')
                font['name'].names = []
            set_font_names(font, new_family_name, new_style_name)

        font.save(output_path)
    
    finally:
        if font:
            font.close()

# Example usage
if __name__ == "__main__":
    folder_path = "predicted characters"
    prepared_char_image_list = []
    for file_name in os.listdir(folder_path):
        if not file_name.endswith(".png"):
            continue
        char_val = file_name[5]
        image = cv2.imread(os.path.join(folder_path, file_name), cv2.COLOR_BGR2GRAY)
        prepared_char_image_list.append((char_val, image))
    
    update_font_from_images(
        font_path="FINAL/arial.ttf",
        char_image_list = prepared_char_image_list,
        output_path="FINAL/my_font.ttf",
        new_family_name="My Handwriting",
        new_style_name="Regular"
    )