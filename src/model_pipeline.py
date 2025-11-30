"""
Modular pipeline building for different sports and tasks.
"""
from typing import Dict, List, Any
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, mean_absolute_error,
    r2_score, roc_auc_score, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor


def build_pipeline(sport_features: Dict[str, List[str]], task: str, hyperparameters: Dict[str, Any] = None) -> Pipeline:
    """
    Build a scikit-learn pipeline for a given sport and task.

    Args:
        sport_features: Dict with keys 'categorical', 'boolean', 'numeric'
        task: 'classification' or 'regression'
        hyperparameters: Optional dict of model hyperparameters

    Returns:
        Configured sklearn Pipeline
    """
    if hyperparameters is None:
        hyperparameters = {}

    categorical_features = sport_features.get('categorical', [])
    boolean_features = sport_features.get('boolean', [])
    numeric_features = sport_features.get('numeric', [])

    preprocessor = ColumnTransformer(
        transformers=[
            # Categorical features
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             [c for c in categorical_features if c is not None]),
            # Boolean features
            ('bool', OneHotEncoder(handle_unknown='ignore', drop=None, sparse_output=False),
             [b for b in boolean_features if b is not None]),
            # Numeric features with imputation and scaling
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), [n for n in numeric_features if n is not None])
        ],
        remainder='drop'
    )

    # Default hyperparameters
    n_estimators = int(hyperparameters.get('n_estimators', 200))
    max_depth = hyperparameters.get('max_depth')
    if max_depth is not None:
        max_depth = int(max_depth)
    
    if task == 'classification':
        from sklearn.ensemble import RandomForestClassifier
        class_weight = hyperparameters.get('class_weight', 'balanced')
        # Handle 'None' string from UI
        if class_weight == 'None':
            class_weight = None

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1
        )
        pipeline = Pipeline(steps=[('prep', preprocessor), ('clf', model)])
    elif task == 'regression':
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        pipeline = Pipeline(steps=[('prep', preprocessor), ('reg', model)])
    else:
        raise ValueError("task must be 'classification' or 'regression'")

    return pipeline


def evaluate_model(pipeline: Pipeline, X_test: Any, y_test: Any, task: str) -> Dict[str, Any]:
    """
    Evaluate a trained model and return metrics.

    Args:
        pipeline: Trained sklearn pipeline
        X_test: Test features
        y_test: Test targets
        task: 'classification' or 'regression'

    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {'task': task}

    if task == 'classification':
        y_pred = pipeline.predict(X_test)
        y_prob = None
        try:
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        except Exception:
            pass

        metrics['accuracy'] = float(accuracy_score(y_test, y_pred))
        metrics['report'] = classification_report(y_test, y_pred, output_dict=False)

        if y_prob is not None:
            # Calibration summary
            metrics['pred_prob_mean'] = float(np.mean(y_prob))
            # ROC-AUC
            try:
                metrics['roc_auc'] = float(roc_auc_score(y_test, y_prob))
            except Exception:
                pass

        # Confusion matrix
        try:
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                metrics['confusion_matrix'] = {
                    'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
                }
        except Exception:
            pass

        # Feature Importance (Random Forest Classifier)
        try:
            model = pipeline.named_steps['clf']
            preprocessor = pipeline.named_steps['prep']
            
            if hasattr(model, 'feature_importances_'):
                feature_names = preprocessor.get_feature_names_out()
                importances = model.feature_importances_
                
                # Create list of (feature, importance) tuples
                importance = []
                for name, val in zip(feature_names, importances):
                    importance.append({'feature': name, 'importance': float(val)})
                
                # Sort by importance
                importance.sort(key=lambda x: x['importance'], reverse=True)
                metrics['feature_importance'] = importance[:20]  # Top 20
        except Exception as e:
            print(f"Error extracting feature importance: {e}")
            pass

    else:  # regression
        y_pred = pipeline.predict(X_test)
        metrics['mae'] = float(mean_absolute_error(y_test, y_pred))
        metrics['r2'] = float(r2_score(y_test, y_pred))

        # Feature Importance (Random Forest)
        try:
            model = pipeline.named_steps['reg']
            preprocessor = pipeline.named_steps['prep']
            
            if hasattr(model, 'feature_importances_'):
                feature_names = preprocessor.get_feature_names_out()
                importances = model.feature_importances_
                
                # Create list of (feature, importance) tuples
                importance = []
                for name, val in zip(feature_names, importances):
                    importance.append({'feature': name, 'importance': float(val)})
                
                # Sort by importance
                importance.sort(key=lambda x: x['importance'], reverse=True)
                metrics['feature_importance'] = importance[:20]  # Top 20
        except Exception as e:
            print(f"Error extracting feature importance: {e}")
            pass

    return metrics
