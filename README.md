
<div align="center">
  
[1]: https://github.com/Pradnya1208
[2]: https://www.linkedin.com/in/pradnya-patil-b049161ba/
[3]: https://public.tableau.com/app/profile/pradnya.patil3254#!/
[4]: https://twitter.com/Pradnya1208


[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/e257c5d6cf02f13072429935b0828525c601414f/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c9f9c5dc4e24eff0143b3056708d24650cbccdde/icons/iconmonstr-twitter-5.svg)][4]

</div>

# <div align="center">Phân loại bệnh tiểu đường sử dụng KNN</div>

<div align="center">
<img src  ="https://github.com/Pradnya1208/Diabetes-classification-using-KNN/blob/main/output/AboutDiabetes.jpg?raw=true" width="100%">
</div>

## Mục tiêu:
Trong dự án này, mục tiêu của chúng tôi là dự đoán khả năng mắc bệnh tiểu đường dựa trên các thông số chẩn đoán.

## Dữ liệu:
[Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database)

### Thông tin về dữ liệu:
Bộ dữ liệu này xuất phát từ Viện Quốc gia về Bệnh Tiểu đường và Bệnh Tiêu hóa & Thận. Mục tiêu của bộ dữ liệu là chẩn đoán dự đoán liệu một bệnh nhân có mắc bệnh tiểu đường hay không, dựa trên một số thông số chẩn đoán trong bộ dữ liệu. Các giới hạn cụ thể đã được đặt ra để chọn những trường hợp này từ một cơ sở dữ liệu lớn hơn. Đặc biệt, tất cả bệnh nhân đều là nữ, ít nhất 21 tuổi và thuộc dân tộc Pima Indian.<br>
Bộ dữ liệu bao gồm các biến dự đoán y tế và một biến mục tiêu là `Outcome`. Các biến dự đoán bao gồm số lần mang thai, chỉ số BMI, mức insulin, tuổi tác, v.v.

## Triển khai:

**Thư viện sử dụng:** `sklearn` `Matplotlib` `pandas` `seaborn` `NumPy` `Scipy` 


## Một số khám phá dữ liệu (EDA):
### Đặc điểm của dữ liệu:
![Features1](https://github.com/Pradnya1208/Diabetes-classification-using-KNN/blob/main/output/eda1.PNG?raw=true)
![Features2](https://github.com/Pradnya1208/Diabetes-classification-using-KNN/blob/main/output/eda2.PNG?raw=true)


## Điền dữ liệu thiếu:
```python
diabetes_data_copy['Glucose'].fillna(diabetes_data_copy['Glucose'].mean(), inplace = True)
diabetes_data_copy['BloodPressure'].fillna(diabetes_data_copy['BloodPressure'].mean(), inplace = True)
diabetes_data_copy['SkinThickness'].fillna(diabetes_data_copy['SkinThickness'].median(), inplace = True)
diabetes_data_copy['Insulin'].fillna(diabetes_data_copy['Insulin'].median(), inplace = True)
diabetes_data_copy['BMI'].fillna(diabetes_data_copy['BMI'].median(), inplace = True)
```
### Vẽ đồ thị sau khi điền dữ liệu:
![Imputed data1](https://github.com/Pradnya1208/Diabetes-classification-using-KNN/blob/main/output/eda_nan1.PNG?raw=true)
![Imputed data2](https://github.com/Pradnya1208/Diabetes-classification-using-KNN/blob/main/output/eda_nan2.PNG?raw=true)



## Huấn luyện mô hình và đánh giá:

### KNN
```python
for i in range(1,15):

    knn = KNeighborsClassifier(i)
    knn.fit(X_train,y_train)
    
```
```
Max test score 76.5625 % và k = [11]
```
![Result](https://github.com/Pradnya1208/Diabetes-classification-using-KNN/blob/main/output/traintestscore.PNG?raw=true)

### Vẽ vùng quyết định:
![Decision Regions](https://github.com/Pradnya1208/Diabetes-classification-using-KNN/blob/main/output/Decision%20regions.PNG?raw=true)


### Ma trận nhầm lẫn:
Ma trận nhầm lẫn là kỹ thuật tóm tắt hiệu suất của một thuật toán phân loại, tức là có đầu ra nhị phân.
![CM](https://github.com/Pradnya1208/Diabetes-classification-using-KNN/blob/main/output/confusion%20matrix.PNG?raw=true)<br>

**Kết quả:**<br>
<img src = "https://github.com/Pradnya1208/Diabetes-classification-using-KNN/blob/main/output/cMresults.PNG?raw=true">   
<img src = "https://github.com/Pradnya1208/Diabetes-classification-using-KNN/blob/main/output/plotcm.PNG?raw=true">

### Báo cáo phân loại:
> **Precision**: Tỷ lệ giữa số dự đoán đúng thuộc lớp dương tính trên tổng số dự đoán là dương tính.

Precision = TP/TP+FP

> **Recall**: Tỷ lệ giữa số dự đoán đúng thuộc lớp dương tính trên tổng số thực sự là dương tính.

Recall = TP/TP+FN

> **F1 Score**: Trung bình điều hòa của Precision và Recall.

F1 Score = 2(Recall Precision) / (Recall + Precision)

![classificationreport](https://github.com/Pradnya1208/Diabetes-classification-using-KNN/blob/main/output/classification%20report.PNG?raw=true)

### Đường cong ROC-AUC:
ROC (Receiver Operating Characteristic) cho biết khả năng phân biệt giữa hai lớp của mô hình.<br>
![rocauc](https://github.com/Pradnya1208/Diabetes-classification-using-KNN/blob/main/output/rocauc.PNG?raw=true)



## Tối ưu hóa

* **Scaling:** 
Quan trọng để chuẩn hóa tất cả các biến trước khi áp dụng thuật toán dựa trên khoảng cách như KNN.

* **Cross Validation:**
Kỹ thuật tránh overfitting hoặc underfitting khi chia dữ liệu thành tập huấn luyện và kiểm tra.

* **Tìm tham số tối ưu:** Sử dụng Grid Search.
```python
from sklearn.model_selection import GridSearchCV
parameters_grid = {"n_neighbors": np.arange(0,50)}
knn= KNeighborsClassifier()
knn_GSV = GridSearchCV(knn, param_grid=parameters_grid, cv = 5)
knn_GSV.fit(X, y)
print("Best Params" ,knn_GSV.best_params_)
print("Best score" ,knn_GSV.best_score_)
```
```
Best Params {'n_neighbors': 25}
Best score 0.7721840251252015
```

### Bài học kinh nghiệm:
`Điền dữ liệu thiếu` `Xử lý giá trị ngoại lai` `Kỹ thuật đặc trưng` `Tối ưu hóa tham số`

## Tham khảo:
- [Skewness](https://www.statisticshowto.datasciencecentral.com/probability-and-statistics/skewed-distribution/)
- [Scaling](https://medium.com/@rrfd/standardize-or-normalize-examples-in-python-e3f174b65dfc)
- [Confusion Matrix](https://medium.com/@djocz/confusion-matrix-aint-that-confusing-d29e18403327)
- [Báo cáo phân loại](http://joshlawman.com/metrics-classification-report-breakdown-precision-recall-f1/)

### Phản hồi:
Nếu bạn có bất kỳ góp ý nào, vui lòng liên hệ tại **pradnyapatil671@gmail.com**

### 🚀 Về tôi
#### Xin chào, tôi là Pradnya! 👋
Tôi là người đam mê AI, thực hành Data Science & Machine Learning.

[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-pred

iction/e257c5d6cf02f13072429935b0828525c601414f/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c9f9c5dc4e24eff0143b3056708d24650cbccdde/icons/iconmonstr-twitter-5.svg)][4]

- 🚀 Đang học: `Machine Learning` `Deep Learning` `Reinforcement learning`
