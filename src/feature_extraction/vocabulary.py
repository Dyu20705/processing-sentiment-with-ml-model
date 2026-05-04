import collections
#mapping word -> index
class Vocabulary:
    #dict = {"<UNK>": 0, "world": 1, "Hello": 2}
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.freeze = False

    def reverse_vocab(self):
        #Tạo dict ngược lại để lấy từ từ index
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def add(self, word):
        if self.freeze:
            print("Vocab đã bị đóng, không thể thêm từ mới.")

        lenDict = len(self.word2idx)
        if word not in self.word2idx:
            self.word2idx[word] = lenDict

    def get_index(self, word):
        #Lấy index của từ, nếu không có trả về index của <UNK>
        return self.word2idx.get(word, self.word2idx.get('<UNK>'))

    def get_word(self, index):
        return self.idx2word.get(index, self.idx2word.get(0))
    
    def freeze(self):
        #Không cho thêm từ mới vào vocab
        self.freeze = True
    
    def build_from_corpus(self, words, max_features=None):
        if self.freeze:
            print("Vocab đã bị đóng, không thể xây dựng từ corpus.")
            return
        
        #Đếm số lần xuất hiện của mỗi từ
        word_count = collections.Counter(words)

        #Lấy n từ phổ biến nhất (n = max_features)
        most_common = word_count.most_common(max_features)
        
        #Tạo vocab từ các từ phổ biến nhất
        for word, _ in most_common:
            self.add(word)

        #Cập nhật reverse vocab
        self.reverse_vocab()