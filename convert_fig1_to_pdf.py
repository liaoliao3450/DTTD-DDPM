import os
from PIL import Image


def convert_png_to_pdf(png_path: str, pdf_path: str):
    """将单张 PNG 转为 PDF（适合论文插图使用）"""
    img = Image.open(png_path).convert("RGB")
    img.save(pdf_path, "PDF", resolution=300.0)
    print(f"[OK] Saved: {pdf_path}")


def main():
    base_dir = os.path.join("paper_results", "figures")

    for name in ["fig1a", "fig1b"]:
        png_path = os.path.join(base_dir, f"{name}.png")
        pdf_path = os.path.join(base_dir, f"{name}.pdf")

        if not os.path.exists(png_path):
            print(f"[WARN] PNG not found: {png_path}")
            continue

        convert_png_to_pdf(png_path, pdf_path)


if __name__ == "__main__":
    main()
