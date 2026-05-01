import re
#normalize text, tokenize, remove stop words, etc.
class TextProcessor:
    def clean(self, text):
        #Xóa HTML tags
        text = re.sub(r'<.*?>', '', text)

        #Xóa ký tự đặc biệt và số
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        return text.lower() #Chuyển về chữ thường
    
    def tokenize(self, text):
        return text.split() #Tách thành list các từ
    
    def process(self, text):
        text = self.clean(text)
        tokens = self.tokenize(text)
        return tokens