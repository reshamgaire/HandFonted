import argparse
import os
import sys

from character_classification import classify_characters
from character_segmentation import segment_characters
from font_creation import update_font_from_images

def main():
    """
    Main function to run the full handwriting-to-font pipeline.
    Parses command-line arguments to configure the process.
    """
    parser = argparse.ArgumentParser(
        description="""
        HandFonted: A tool to generate a TrueType Font from an image of handwritten characters.
        The process involves three main steps:
        1. Segmenting characters from the source image.
        2. Classifying each character using a pre-trained model.
        3. Building the final .ttf font file with the classified character images.
        """,
        formatter_class=argparse.RawTextHelpFormatter  # Allows for better formatting of the description
    )

    # --- Input and Output Arguments ---
    parser.add_argument(
        '--input-image',
        type=str,
        required=True,
        help='Path to the input image of handwritten characters.'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        required=True,
        help='Path to save the final generated .ttf font file (e.g., "output/my_font.ttf").'
    )
    
    # --- Model and Font Resource Arguments ---
    parser.add_argument(
        '--model-path',
        type=str,
        default="resources/best_ResInceptionNet_model0.8811.pth",
        help='Path to the pre-trained character classification model (.pth file).'
    )
    parser.add_argument(
        '--base-font',
        type=str,
        default="resources/arial.ttf",
        help='Path to a base .ttf font file (e.g., Arial.ttf) to use for character metrics and as a template.'
    )

    # --- Font Customization Arguments ---
    parser.add_argument(
        '--font-name',
        type=str,
        default='My Handwriting',
        help='The family name for the new font (e.g., "Jane Doe Hand"). Default: "My Handwriting".'
    )
    parser.add_argument(
        '--font-style',
        type=str,
        default='Regular',
        help='The style name for the new font (e.g., "Regular", "Bold"). Default: "Regular".'
    )
    parser.add_argument(
        '--thickness',
        type=int,
        default=100,
        help='Desired stroke thickness for the font glyphs. Adjust for a bolder or lighter look. Default: 100.'
    )
    
    # --- Intermediate File Arguments ---
    parser.add_argument(
        '--intermediate-dir',
        type=str,
        default='',
        help='Directory to save the intermediate classified character images.'
    )

    args = parser.parse_args()

    # --- Ensure output directories exist ---
    # For the final font file
    output_dir = os.path.dirname(args.output_path)
    if output_dir: # Check if the path contains a directory part
        os.makedirs(output_dir, exist_ok=True)
    
    # For the intermediate character images (the classification function also does this, but it's good practice)
    os.makedirs(args.intermediate_dir, exist_ok=True)


    # --- Start the Pipeline ---
    print(f"🚀 Starting HandFonted pipeline...")
    print(f"[*] Input Image: {args.input_image}")

    # 1. Segment Characters
    print("\n[Step 1/3] Segmenting characters from image...")
    list_of_char_images = segment_characters(args.input_image)

    if not list_of_char_images:
        print("\n❌ Error: No characters were segmented from the image. Please check the input image.")
        sys.exit(1) # Exit with an error code
    
    print(f"✅ Found {len(list_of_char_images)} potential characters.")

    # 2. Classify Characters
    print("\n[Step 2/3] Classifying segmented characters...")
    char_images = classify_characters(
        list_of_char_images, 
        model_path=args.model_path, 
        OUTPUT_PATH=args.intermediate_dir
    )
    
    if not char_images:
        print("\n❌ Error: Character classification failed to produce any valid characters.")
        sys.exit(1)

    print(f"✅ Classified {len(char_images)} characters. Intermediate images saved to '{args.intermediate_dir}/'.")

    # 3. Create Font
    print("\n[Step 3/3] Generating the font file...")
    update_font_from_images(
        font_path=args.base_font,
        char_image_list=char_images,
        output_path=args.output_path,
        desired_thickness=args.thickness,
        new_family_name=args.font_name,
        new_style_name=args.font_style
    )

    print(f"\n🎉 Success! Font '{args.font_name}' was created successfully.")
    print(f"[*] Your new font is saved at: {os.path.abspath(args.output_path)}")


if __name__ == "__main__":
    main()