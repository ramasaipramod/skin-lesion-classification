# Skin Lesion Classification

This repository contains code and resources for a machine learning project aimed at classifying skin lesions from dermatoscopic images. The project leverages feature extraction techniques, various classification models, and data balancing strategies to improve diagnostic accuracy.

## Files in the Repository

- **ModelPerformances.ipynb**  
  Contains the evaluation of various machine learning models, including SVM, Random Forest, XGBoost, and CNNs, with detailed performance metrics such as precision, recall, and F1-score.

- **Preprocessing.ipynb**  
  Handles data preprocessing, including feature extraction techniques such as Gabor filters, wavelet transforms, and edge detection. This notebook prepares the dataset for training and testing.

- **SkinLesionClassification.ipynb**  
  Implements skin lesion classification using custom sampling techniques and multiple machine learning models.

- **SkinLesionClassification_Using_SMOTETomek.ipynb**  
  Explores the impact of using SMOTETomek for data balancing and evaluates its influence on classification performance.

## Dataset

The dataset comprises 10,015 dermatoscopic images, representing seven skin lesion categories:
- Actinic keratoses and intraepithelial carcinoma (akiec)
- Basal cell carcinoma (bcc)
- Benign keratosis-like lesions (bkl)
- Dermatofibroma (df)
- Melanoma (mel)
- Melanocytic nevi (nv)
- Vascular lesions (vasc)

The dataset is sourced from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T).

## Features and Models

### Feature Extraction
- **Gabor Filters**: Texture analysis.
- **Wavelet Transform**: Sparse image representation.
- **Mean Color and Edges**: Color composition and lesion shape analysis.

### Classification Models
- Support Vector Machines (SVM)
- Random Forest
- XGBoost
- Feedforward Neural Networks (FNN)
- Convolutional Neural Networks (CNN)

## Results and Analysis
- Custom sampling techniques were compared to SMOTETomek for data balancing.
- XGBoost coupled with wavelet transforms showed the best performance.
- The repository includes confusion matrices and performance metrics for all approaches.

## References
- [Harvard Dataverse Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
- [scikit-learn](https://scikit-learn.org)
- [imbalanced-learn](https://imbalanced-learn.org)
- [OpenCV](https://opencv.org)
