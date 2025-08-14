import cv2 as cv
import os

def convert_tiff_to_png(tiff_path, png_path):
    """
    Convert a TIFF image to PNG format.
    
    :param tiff_path: Path to the input TIFF file.
    :param png_path: Path where the output PNG file will be saved.
    """
    # Read the TIFF image
    image = cv.imread(tiff_path, cv.IMREAD_UNCHANGED)

    # Save the image as PNG
    cv.imwrite(png_path, image)
    print(f"Converted {tiff_path} to {png_path}")

def main():
    input_word = os.path.join("/mnt/UGent_Share/ghentcdh_belhisfirm/Source/Test_Scans_Iguana/TEST UAntwerpen_31072025_400ppi", input("Enter the name to the TIFF file: "))
    input_name = input_word.split("/")[-1].split(".")[0]
    output = f"experiments/LabelStudio/png_images/{input_name}.png"
    convert_tiff_to_png(input_word, output)

# Run the main function
if __name__ == "__main__":
    main()