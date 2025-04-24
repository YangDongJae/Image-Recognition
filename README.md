Image Recognition and Retrieval Pipeline - Caltech20

This repository provides a complete end-to-end image classification and retrieval pipeline using the Caltech20 dataset. It leverages both classical and deep learning-based feature extractors and compares the performance across multiple classifiers with quantitative and qualitative analysis.

Place the Caltech20 dataset in the following directory structure:

Caltech20/
  ├── class1/
  │     ├── img1.jpg
  │     ├── img2.jpg
  ├── class2/
        ├── img1.jpg
        ├── img2.jpg

Requirements

Install the required packages with:

pip install -r requirements.txt

Make sure to manually install these key packages if needed:

opencv-python

Pillow

scikit-learn

torch, torchvision

imbalanced-learn

🚀 Running the Pipeline

1. Data Preparation

Load and split images into training/testing sets using Caltech20DatasetSplitter

Transform and normalize images for model input

2. Feature Extraction

Bag-of-Words (BoW) using Dense SIFT and KMeans

Spatial Pyramid Matching

VGG13 and VGG19 Deep Features (via torchvision.models)

3. Classifier Training & Evaluation

Three classifiers used:

Linear SVM

Random Forest

2-layer Fully Connected Network (PyTorch)

Confusion matrices and accuracy comparisons are generated

4. Image Retrieval

Perform top-5 retrieval based on feature similarity

Visualize retrieval results with color-coded labels

5. Grad-CAM & Activation Visualization

Visualize feature maps across layers in VGG13 and VGG16

Generate heatmaps highlighting key image regions

6. Advanced Visualization

Visual word distributions using PCA and KMeans

Histograms of visual words per image or spatial region

t-SNE plots of query/retrieval relationships

🧪 Experimental Extensions

✅ Hyperparameter Tuning

GridSearchCV is used for SVM and Random Forest tuning

✅ Feature Enhancement

Standardization and PCA applied to improve classifier performance

✅ Class Imbalance Handling

SMOTE and Hybrid Resampling for balanced class training

✅ Improved Classifier

A flexible, deeper ImprovedFCClassifier using LayerNorm and Dropout

Early stopping and ReduceLROnPlateau scheduler included

📊 Result Visualizations

classification_accuracy_heatmap.png: Heatmap of classification accuracy

retrieval_results.png: Retrieval examples from top-5 most similar images

gradcam_visualization.png: Grad-CAM heatmaps for VGG models

spatial_pyramid_hierarchy.png: Spatial BoW representation breakdown

confusion_matrices.png: All classifier confusion matrices

retrieval_precision_boxplot.png: Boxplot showing retrieval accuracy distribution

🏁 Final Notes

You can tweak hyperparameters and feature extractors in the script

Additional classifier types (e.g., XGBoost) or datasets can be plugged in

For best performance, run on GPU-supported environment (CUDA or MPS)

📬 Contact

For improvements or questions, please open an issue or contact the contributor.

