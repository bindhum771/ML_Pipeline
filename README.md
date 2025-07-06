# ML Pipeline Project

A simple end-to-end machine learning pipeline using Pandas and Scikit-learn:
✅ Data loading  
✅ Preprocessing (impute, scale, encode)  
✅ Train Logistic Regression  
✅ Plot confusion matrix  
✅ Save & reuse trained model

## 📂 Structure
- `pipeline.py`: main script
- `data/cleaned_pipeline_file.csv`: dataset
- `outputs/confusion_matrix.png`: saved plot
- `models/trained_model_pipeline.pkl`: saved model

## 🚀 How to run
```
pip install -r requirements.txt
python pipeline.py
```

## 📊 Output
- Prints accuracy
- Saves confusion matrix
- Saves trained model for reuse