"""
Indian Medicinal Plant Leaf Identification (PS-10) Source Package.
"""
from src.data import (
    CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS, SPECIES_METADATA,
    scan_dataset, compute_perceptual_hashes, create_leak_free_splits,
    verify_leakage, compute_train_normalization, get_transforms,
    MedicinalLeafDataset, PreloadedMedicinalLeafDataset, mixup_cutmix_collate_fn
)
from src.models import (
    CustomCNNBaseline, build_resnet50, build_efficientnet_b0,
    build_mobilenet_v3_large, EnsembleClassifier, get_model, count_parameters
)
from src.train import fit_model, evaluate_epoch
from src.evaluate import (
    evaluate_test_set, plot_confusion_matrix, find_top_confused_pairs,
    compute_ece, TemperatureScaler, compute_rejection_curve, evaluate_robustness
)
from src.gradcam import generate_gradcam_overlay, plot_explainability_grid
