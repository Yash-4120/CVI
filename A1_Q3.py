import cv2
import numpy as np
import matplotlib.pyplot as plt

# Store operation history
history = []
image_stack = []


def show_preview(original, edited, title):
    """Show side-by-side comparison using matplotlib."""
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(edited, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def adjust_brightness(img, value):
    """Increase or decrease brightness."""
    result = cv2.convertScaleAbs(img, alpha=1, beta=value)
    history.append(f"Brightness {value:+d}")
    return result


def adjust_contrast(img, factor):
    """Adjust image contrast."""
    result = cv2.convertScaleAbs(img, alpha=factor, beta=0)
    history.append(f"Contrast ×{factor}")
    return result


def to_grayscale(img):
    """Convert image to grayscale (still 3-channel for consistency)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    history.append("Grayscale")
    return result


def add_padding(img, pad_size, border_type):
    """Add border/padding around the image."""
    border_types = {
        "constant": cv2.BORDER_CONSTANT,
        "reflect": cv2.BORDER_REFLECT,
        "replicate": cv2.BORDER_REPLICATE
    }
    if border_type not in border_types:
        border_type = "constant"
    result = cv2.copyMakeBorder(img, pad_size, pad_size, pad_size, pad_size,
                                border_types[border_type])
    history.append(f"Padded {pad_size}px ({border_type})")
    return result


def apply_threshold(img, mode):
    """Apply binary or inverse threshold."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh_type = cv2.THRESH_BINARY if mode == "binary" else cv2.THRESH_BINARY_INV
    _, th = cv2.threshold(gray, 127, 255, thresh_type)
    result = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    history.append(f"Threshold ({mode})")
    return result


def blend_images(img, second_path, alpha):
    """Blend current image with another image using manual alpha."""
    second = cv2.imread(second_path)
    if second is None:
        print(" Could not read second image.")
        return img
    second = cv2.resize(second, (img.shape[1], img.shape[0]))
    blended = cv2.addWeighted(img, alpha, second, 1 - alpha, 0)
    history.append(f"Blended (alpha={alpha})")
    return blended


def undo_last():
    """Undo last operation."""
    if len(image_stack) > 1:
        image_stack.pop()
        history.append("Undo")
        return image_stack[-1].copy()
    else:
        print("⚠️ Nothing to undo.")
        return image_stack[-1]


def photo_editor():
    """Main interactive menu loop."""
    path = input("Enter image file path: ").strip()
    img = cv2.imread(path)
    if img is None:
        print(" Image not found. Please check the path.")
        return

    original = img.copy()
    image_stack.append(img.copy())

    while True:
        print("""
==== Mini Photo Editor ====
1. Adjust Brightness
2. Adjust Contrast
3. Convert to Grayscale
4. Add Padding (choose border type)
5. Apply Thresholding (binary or inverse)
6. Blend with Another Image
7. Undo Last Operation
8. View History
9. Save and Exit
""")

        choice = input("Select option (1-9): ").strip()

        if choice == "1":
            value = int(input("Brightness value (-100 to 100): "))
            img = adjust_brightness(img, value)
        elif choice == "2":
            factor = float(input("Contrast factor (e.g., 1.2): "))
            img = adjust_contrast(img, factor)
        elif choice == "3":
            img = to_grayscale(img)
        elif choice == "4":
            pad = int(input("Padding size (pixels): "))
            btype = input("Border type (constant/reflect/replicate): ").lower()
            img = add_padding(img, pad, btype)
        elif choice == "5":
            mode = input("Threshold type (binary/inverse): ").lower()
            img = apply_threshold(img, mode)
        elif choice == "6":
            path2 = input("Enter second image path: ").strip()
            alpha = float(input("Alpha (0-1): "))
            img = blend_images(img, path2, alpha)
        elif choice == "7":
            img = undo_last()
        elif choice == "8":
            print("\n=== Operation History ===")
            for h in history:
                print("•", h)
            print()
        elif choice == "9":
            save_name = input("Enter filename to save (e.g., output.jpg): ")
            cv2.imwrite(save_name, img)
            print(" Image saved. Exiting.")
            break
        else:
            print("Invalid choice. Try again.")
            continue

        image_stack.append(img.copy())
        show_preview(original, img, "Preview")

if __name__ == "__main__":
    photo_editor()
