# Hướng dẫn tự xây mô hình sentiment analysis từ đầu

Tài liệu này dành cho dự án phân loại cảm xúc với bộ dữ liệu Stanford Sentiment Treebank trên Kaggle. Mục tiêu không phải là đưa sẵn code hoàn chỉnh, mà là giúp bạn hiểu đúng quy trình, tự hiện thực từng bước, và biết vì sao từng mô hình hoạt động.

## 1. Bài toán và mục tiêu

Bộ dữ liệu này là bài toán phân loại nhị phân: câu văn mang cảm xúc tích cực hay tiêu cực. Trên Kaggle, bài toán được đánh giá bằng ROC-AUC, nên khi huấn luyện bạn cần quan tâm đến xác suất dự đoán, không chỉ nhãn 0 hoặc 1.

Điều quan trọng nhất khi học bài toán này là xây được một pipeline ổn định:

- Nạp dữ liệu đúng cách
- Làm sạch văn bản nhất quán
- Chuyển văn bản thành đặc trưng số
- Huấn luyện nhiều mô hình trên cùng một tập chia
- So sánh bằng cùng một metric
- Phân tích lỗi để hiểu vì sao mô hình sai

## 2. Cách tải dữ liệu từ Kaggle

Đây là competition Stanford Sentiment Treebank, không phải dataset IMDB. Nếu bạn muốn làm đúng theo bài toán Kaggle, hãy tải dữ liệu từ competition này.

Các bước khuyến nghị:

1. Tạo tài khoản Kaggle và đăng nhập.
2. Vào trang competition và nhấn Accept Rules nếu được yêu cầu.
3. Tạo API token trong phần Account của Kaggle, tải file kaggle.json.
4. Cài Kaggle CLI trên máy.
5. Đặt kaggle.json vào thư mục cấu hình Kaggle mặc định của máy.
6. Tải dữ liệu competition về máy và giải nén vào một thư mục dữ liệu riêng trong project.

Nếu bạn muốn giữ mọi thứ đơn giản, hãy tạo một thư mục như data/ với các file gốc, file train, validation, test và bất kỳ metadata nào Kaggle cung cấp. Không nên trộn dữ liệu đã xử lý vào cùng thư mục gốc với dữ liệu thô.

Lưu ý quan trọng:

- Không dùng test để tinh chỉnh mô hình.
- Chỉ dùng validation để chọn đặc trưng, siêu tham số và ngưỡng dự đoán.
- Nếu competition yêu cầu file submission, hãy luôn giữ đúng định dạng id và target xác suất.

## 3. Luồng làm việc hiệu quả

Cách học và làm hiệu quả nhất là đi từ đơn giản đến phức tạp.

### Bước 1: Xây baseline trước

Bắt đầu bằng mô hình dễ hiểu nhất, thường là Naive Bayes với bag-of-words. Mục tiêu của baseline không phải là tốt nhất, mà là để kiểm tra pipeline có chạy đúng không.

### Bước 2: Giữ chung một bộ tiền xử lý

Tất cả mô hình nên dùng chung logic xử lý văn bản, chỉ khác ở cách biểu diễn đặc trưng. Như vậy bạn sẽ so sánh công bằng hơn và hiểu ảnh hưởng của từng thành phần.

### Bước 3: Đánh giá theo cùng một chuẩn

Nên theo dõi:

- ROC-AUC cho Kaggle
- Accuracy để nhìn trực quan
- Precision, Recall, F1 để hiểu lỗi lệch lớp
- Confusion matrix để xem mô hình sai ở đâu

### Bước 4: Phân tích lỗi

Sau mỗi lần huấn luyện, hãy xem các mẫu sai điển hình:

- Câu có phủ định như not good
- Câu mỉa mai
- Câu rất dài hoặc quá ngắn
- Từ mới không có trong từ điển
- Câu có nhiều từ trung tính nhưng nhãn vẫn rõ ràng

Phân tích lỗi là cách nhanh nhất để hiểu mô hình sâu hơn thay vì chỉ nhìn điểm số.

## 4. Cấu trúc tư duy cho dự án này

Repo hiện tại đang có các phần chính sau:

- preprocessing/text_processor.py: xử lý văn bản thô, chuẩn hóa, tách từ
- feature/vocabulary.py: quản lý từ vựng và ánh xạ từ sang chỉ số
- feature/feature_extraction.py: chuyển câu thành vector số
- models/naive_bayes.py: nơi hiện thực Naive Bayes
- main.py: nơi chạy pipeline train và evaluate
- utils.py: các hàm hỗ trợ chung như chia dữ liệu, tính metric, ghi kết quả

Nếu bạn muốn tự phát triển lâu dài, hãy giữ nguyên nguyên tắc này:

- preprocessing chỉ xử lý text
- feature chỉ xử lý biểu diễn số
- model chỉ học từ đặc trưng số
- main chỉ điều phối luồng chạy

Nhờ vậy mỗi phần dễ thay thế và dễ kiểm thử.

## 5. Cách xây từng mô hình từ scratch

### 5.1 Naive Bayes

Đây là mô hình nên làm đầu tiên vì nó dạy bạn tư duy xác suất và bag-of-words.

Bạn cần hiểu ba ý chính:

- Mỗi câu được xem như tập hoặc chuỗi từ độc lập có điều kiện theo nhãn.
- Xác suất của một câu được tính từ xác suất của từng từ trong nhãn.
- Cần làm trơn Laplace để tránh xác suất bằng 0 khi gặp từ chưa thấy.

Cách học tốt nhất với Naive Bayes là:

- Dùng bag-of-words trước, chưa cần n-gram phức tạp.
- So sánh binary features và count features.
- Luôn làm việc ở không gian log để tránh tràn số.

Mô hình này thường cho kết quả khá ổn trên văn bản, và là nền tảng tốt để hiểu vì sao đặc trưng từ vựng quan trọng.

Điều cần nhớ:

- Naive Bayes rất nhạy với cách tách từ và tiền xử lý.
- Nó thích dữ liệu văn bản nhiều từ lặp lại.
- Nó là baseline tốt nhưng hiếm khi là mô hình cuối cùng tốt nhất.

### 5.2 Decision Tree

Decision Tree không phải lựa chọn mạnh nhất cho text thô, nhưng lại rất tốt để học tư duy ra quyết định theo nhánh.

Với text, bạn cần chuyển câu thành vector số trước, thường là:

- Binary bag-of-words
- Count vector
- TF-IDF nếu muốn ổn định hơn

Điểm cần hiểu sâu:

- Cây quyết định chia dữ liệu dựa trên tiêu chí giảm entropy hoặc Gini.
- Với dữ liệu text có hàng nghìn chiều, cây rất dễ overfit nếu không giới hạn độ sâu.
- Decision Tree thường hoạt động kém hơn các mô hình tuyến tính trên dữ liệu sparse chiều cao.

Cách học hiệu quả:

- Bắt đầu với vector nhỏ và top-k từ phổ biến nhất.
- Giới hạn max depth, min samples split, min samples leaf.
- So sánh cây rất nông với cây rất sâu để thấy hiện tượng overfitting.

Nếu mục tiêu là học sâu mô hình, cây quyết định giúp bạn hiểu rõ trade-off giữa khả năng diễn giải và khả năng tổng quát hóa.

### 5.3 SVM

SVM thường là một trong những mô hình mạnh nhất cho bài toán phân loại văn bản dạng cổ điển.

Điểm quan trọng nhất:

- SVM không cố gắng mô hình hóa xác suất trực tiếp như Naive Bayes.
- Nó tìm siêu phẳng phân tách với biên an toàn lớn nhất.
- Với văn bản, SVM tuyến tính thường hiệu quả hơn SVM phi tuyến vì dữ liệu rất nhiều chiều và sparse.

Cách học nên đi theo thứ tự:

- Dùng bag-of-words hoặc TF-IDF
- Hiểu hinge loss và regularization
- Hiểu tham số C ảnh hưởng thế nào đến biên và overfitting
- So sánh với logistic regression để hiểu sự khác nhau giữa tối ưu biên và tối ưu xác suất

Nếu tự hiện thực từ đầu, hãy chú ý:

- Đầu vào cần vector số ổn định
- Tối ưu thường dựa trên subgradient hoặc gradient descent biến thể
- Chuẩn hóa đặc trưng giúp mô hình học dễ hơn

SVM là mô hình rất đáng học vì nó dạy bạn cách nhìn bài toán dưới góc tối ưu hóa thay vì chỉ đếm từ.

### 5.4 Neural Network

Neural network là bước tiếp theo khi bạn muốn vượt qua đặc trưng thủ công và học biểu diễn tốt hơn.

Nếu xây từ đầu, nên đi từ đơn giản nhất:

- MLP một hoặc hai hidden layers
- Input là bag-of-words hoặc TF-IDF
- Sau đó mới nâng lên embeddings và pooling

Điểm bạn cần hiểu sâu:

- Forward pass tạo ra dự đoán từ đầu vào
- Backpropagation lan truyền sai số ngược lại để cập nhật trọng số
- Khởi tạo trọng số, learning rate và regularization ảnh hưởng rất mạnh đến kết quả

Lộ trình hợp lý:

- Bắt đầu với MLP trên vector text đã có sẵn
- Học cách hoạt động của activation function như ReLU hoặc tanh
- Thêm dropout hoặc early stopping để tránh overfitting
- Sau đó thử embedding layer để học biểu diễn từ

Với text, neural network mạnh hơn khi dữ liệu đủ lớn hoặc khi bạn dùng biểu diễn từ tốt hơn. Nếu dữ liệu nhỏ và chỉ dùng vector rất thô, mô hình này chưa chắc thắng SVM.

## 6. Khi nào nên dùng mô hình nào

Một quy tắc thực tế:

- Muốn baseline nhanh, dễ hiểu: Naive Bayes
- Muốn học cách chia nhánh và diễn giải quyết định: Decision Tree
- Muốn hiệu năng tốt trên text cổ điển: SVM tuyến tính
- Muốn học biểu diễn và backprop: Neural Network

Nếu mục tiêu là làm Kaggle tốt, thứ tự thử nghiệm hợp lý thường là:

- Naive Bayes
- Linear SVM
- Neural Network đơn giản
- Tinh chỉnh feature engineering như n-gram, TF-IDF, stopword handling, normalization

Decision Tree nên được giữ như một bài học và một baseline giải thích, không nên kỳ vọng nó là mô hình tốt nhất cho dữ liệu text chiều cao.

## 7. Những lỗi thường gặp

- Rò rỉ dữ liệu khi fit vocabulary trên cả train và validation
- Dùng test để chọn tham số
- Xóa quá mạnh làm mất dấu hiệu cảm xúc như not, never, no
- Không xử lý từ chưa thấy trong tập huấn luyện
- So sánh mô hình trên các pipeline tiền xử lý khác nhau
- Chỉ nhìn accuracy mà bỏ qua AUC hoặc lỗi theo lớp
- Dùng mô hình phức tạp nhưng không có baseline rõ ràng

## 8. Cách học sâu để dùng lâu dài

Nếu bạn muốn hiểu mô hình để dùng về sau, hãy luôn tự trả lời các câu hỏi sau:

- Đầu vào của mô hình là gì và tại sao phải biểu diễn như vậy
- Mô hình học được điều gì từ dữ liệu
- Mô hình đang tối ưu hàm mục tiêu nào
- Mô hình sai trong những trường hợp nào
- Có thể cải thiện bằng feature, regularization hay dữ liệu không

Học theo cách này sẽ giúp bạn không chỉ làm xong bài toán sentiment analysis, mà còn chuyển được tư duy sang các bài toán phân loại văn bản khác như spam detection, topic classification, toxicity detection và review rating prediction.

## 9. Gợi ý cách phát triển tiếp dự án

- Hoàn thiện TextProcessor để xử lý tiếng Anh ổn định hơn
- Làm Vocabulary và Feature Extraction thật rõ ràng
- Viết từng model riêng biệt với cùng giao diện fit và predict
- Thêm phần đánh giá trong main.py
- Lưu lại kết quả thử nghiệm theo từng cấu hình
- So sánh ảnh hưởng của n-gram, TF-IDF, stopword removal, và min frequency

## 10. Kết luận

Nếu đi đúng trình tự, bạn sẽ không chỉ có một bộ mô hình chạy được, mà còn hiểu vì sao từng mô hình hoạt động, vì sao mô hình nào hợp với dữ liệu văn bản, và khi nào nên chọn mô hình nào trong thực tế.

Mục tiêu cuối cùng của dự án này là xây được một tư duy thực nghiệm rõ ràng: cùng dữ liệu, cùng metric, cùng cách đánh giá, nhưng nhiều mô hình khác nhau để bạn nhìn ra bản chất của từng phương pháp.
