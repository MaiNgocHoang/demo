from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Tạo một bộ dữ liệu multi-label giả lập
X, y = make_multilabel_classification(n_samples=1000, n_features=20, n_classes=5, n_labels=3, random_state=42)

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo mô hình cơ sở là RandomForestClassifier
        base_classifier = RandomForestClassifier(random_state=42)

# Sử dụng MultiOutputClassifier để mở rộng mô hình RandomForest cho multi-label
multi_output_classifier = MultiOutputClassifier(base_classifier, n_jobs=-1)

# Định nghĩa các giá trị tham số cần tìm kiếm với GridSearchCV
param_grid = {
    'estimator__n_estimators': [50, 100, 150],    # Số lượng cây trong RandomForest
    'estimator__max_depth': [5, 10, None],        # Độ sâu của cây
}

# Khởi tạo GridSearchCV
grid_search = GridSearchCV(multi_output_classifier, param_grid, cv=3, verbose=1, n_jobs=-1)

# Huấn luyện mô hình sử dụng GridSearchCV
grid_search.fit(X_train, y_train)

# In ra tham số tốt nhất
print("Best Parameters:", grid_search.best_params_)

# Dự đoán và đánh giá trên tập kiểm tra
y_pred = grid_search.best_estimator_.predict(X_test)
print("Accuracy on Test Set:", accuracy_score(y_test, y_pred))
