# -*- coding: utf-8 -*-

from nlos_methodology.cross_material import run_cross_material_analysis


if __name__ == "__main__":
    outputs = run_cross_material_analysis()

    print("Cross-material analysis completed.")
    for output_name, output_path in outputs.items():
        print(f"{output_name:28s}: {output_path}")

