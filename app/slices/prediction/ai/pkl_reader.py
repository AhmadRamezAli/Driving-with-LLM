import pickle


with open('app/slices/prediction/ai/vqa_test_1k.pkl', 'rb') as f:
    data = pickle.load(f)
print(data)
