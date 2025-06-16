import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import shap

# Test SHAP handling
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

# Test input
test_input = X.iloc[[0]]
print("Test input shape:", test_input.shape)

# SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(test_input)

print("SHAP values type:", type(shap_values))
if isinstance(shap_values, list):
    print("SHAP values list length:", len(shap_values))
    for i, sv in enumerate(shap_values):
        print(f"Class {i} shape:", sv.shape)
else:
    print("SHAP values shape:", shap_values.shape)

print("Expected value:", explainer.expected_value)
print("Prediction:", model.predict(test_input))

# Test creating explanation object
pred_class = model.predict(test_input)[0]
if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
    single_shap_values = shap_values[0, :, pred_class]
else:
    single_shap_values = shap_values[pred_class][0]

single_expected_value = explainer.expected_value[pred_class]

print("Single SHAP values shape:", single_shap_values.shape)
print("Single expected value:", single_expected_value)

# Create explanation object
waterfall_explanation = shap.Explanation(
    values=single_shap_values,
    base_values=single_expected_value,
    data=test_input.values[0],
    feature_names=list(iris.feature_names)
)

print("Explanation object created successfully")
print("Values shape:", waterfall_explanation.values.shape)
print("Base values:", waterfall_explanation.base_values)
