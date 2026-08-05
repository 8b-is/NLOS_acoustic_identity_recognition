# -*- coding: utf-8 -*-

from pathlib import Path

from nlos_methodology.config import FIGURES_ROOT, OUTPUT_ROOT
from nlos_methodology.visualization import draw_confusion_matrix_from_csv

CM_CSV_PATH = Path(
    r"D:\NLOSKimlik\NLOS_identity_recognition_results"
    r"\results"
    r"\TSET_T1_T2_T3_T4_T5_T6_T7_T8_M"
    r"\Fusion_D0"
    r"\RF"
    # r"\LOSO_overall_CM.csv" 
    # r"\LOSO_scene_0_fold_01_CM.csv"
    # r"\LOSO_scene_8_fold_09_CM.csv"
    r"\LOSO_scene_5_fold_06_CM.csv"
)

# Leave as None for the automatic scene/overall title.
FIGURE_TITLE = None

# Leave as None for a filename containing T-set, feature, classifier, and CM.
OUTPUT_FILE_NAME = None

# Leave as None to infer 12 for scene folds and 120 for the overall CM from
# the largest true-class row total.
COLOR_SCALE_MAXIMUM = None

SHOW_FIGURE = True


def main() -> None:
    saved_path = draw_confusion_matrix_from_csv(
        CM_CSV_PATH,
        title=FIGURE_TITLE,
        output_file_name=OUTPUT_FILE_NAME,
        figures_root=FIGURES_ROOT,
        show_figure=SHOW_FIGURE,
        color_scale_maximum=COLOR_SCALE_MAXIMUM,
    )

    print(f"Selected CM       : {CM_CSV_PATH}")
    print(f"Output root       : {OUTPUT_ROOT.resolve()}")
    print(f"Figures directory : {FIGURES_ROOT.resolve()}")
    print(f"Saved figure      : {saved_path.resolve()}")


if __name__ == "__main__":
    main()
